"""动态视觉特效（ti.func）：日冕辉光 + 临边日珥 + 接触融合辉光
+ 引力透镜（致密天体：中子星 / 黑洞）。

引力透镜四件套：
    _weak_deflect  远场解析单次偏折（α = 2R_s/b，弱场 GR 精确值）
    _photon_march  近场逐段测地线积分（表面/吸积盘/视界都在弯曲路径上）
    _disk_emission 吸积盘单次穿越发射（多普勒束流 + 差速旋转丝缕）
    _photon_ring   阴影边缘的光子环亮线（临界撞击参数环带）

由 pipeline.render_scene 调用。多数函数只在少数像素上有非零
贡献，平均每像素开销可忽略；唯 _photon_march 仅在黑洞附近
（撞击参数 < LENS_CUT·R_s）的像素上激活。

注意：本模块函数被 pipeline 的 kernel 调用时，模块级全局常量按
编译时绑定内联（与 render 包其他模块同一约束）。
"""

import taichi as ti

from .context import (C_INV_LIGHT2, DISK_AMP, DISK_IN, DISK_OUT, DISK_SPIN,
                      FUSE_AMP, FUSE_CTR, FUSE_CUT, FUSE_SIG0, FUSE_SIG1,
                      FUSE_WID, GLOW_IN_AMP, GLOW_IN_SIG, GLOW_OUT_AMP,
                      GLOW_OUT_SIG, LENS_CUT, LENS_ESC, LENS_HMAX,
                      LENS_HMIN, LENS_STEPS, PROM_AMP, PROM_DEPTH, PROM_SIG,
                      RING_AMP, RING_R, RING_W, TYPE_BH, TYPE_NS)
from .noise import _fbm3, _sstep, _vmix, _vnoise
from .star_surface import _star_surface
from .state import (fuse_i, fuse_j, fuse_n, lens_k, lens_n, n_body_f,
                    scr_bb, star_gain_f, star_mass_f, star_pos_f,
                    star_rad_f, star_seeds, star_stretch_f, star_axis_f,
                    star_tints, star_type_f)


# ------------------------------------------------------------ 天体几何
#
# 潮汐变形：拉伸天体按旋转椭球求交（长轴沿潮汐轴 star_axis_f，
# 半轴 (r·f, r/√f, r/√f)，体积守恒）；f < 1.02 时退化为纯球路径
#（与历史版本逐位一致）。射线在潮汐轴正交基下分解后按椭球度量
# 解二次方程，解的参数即世界距离 t（缩放只作用于坐标分量）。

@ti.func
def _axis_basis(ax: ti.template()):
    """与潮汐轴 ax 正交的两个单位向量 e2/e3"""
    ref = ti.Vector([0.0, 0.0, 1.0])
    if ti.abs(ax.z) > 0.9:
        ref = ti.Vector([1.0, 0.0, 0.0])
    e2 = ax.cross(ref).normalized()
    e3 = ax.cross(e2)
    return e2, e3


@ti.func
def _body_hit_t(k: ti.i32, o: ti.template(), d: ti.template()) -> ti.f32:
    """射线-天体求交（o/d 为射线原点与单位方向）-> 世界距离 t（未命中 -1）。

    支持潮汐拉伸椭球；死星（半径 0）恒不命中。直射与透镜
    （测地线积分段内）两条路径共用。
    """
    r = star_rad_f[k]
    t_hit = -1.0
    if r > 0.01:
        rel = o - star_pos_f[k]
        f = star_stretch_f[k]
        if f < 1.02:
            b = rel.dot(d)
            c0 = rel.dot(rel) - r * r
            disc = b * b - c0
            if disc > 0.0:
                th = -b - ti.sqrt(disc)
                if th > 1e-6:
                    t_hit = th
        else:
            sa = r * f                       # 长半轴（沿潮汐轴）
            sb = r / ti.sqrt(f)              # 短半轴（体积守恒）
            ax = star_axis_f[k]
            e2, e3 = _axis_basis(ax)
            ox = rel.dot(ax)
            oy = rel.dot(e2)
            oz = rel.dot(e3)
            dx = d.dot(ax)
            dy = d.dot(e2)
            dz = d.dot(e3)
            ia = 1.0 / (sa * sa)
            ib = 1.0 / (sb * sb)
            A = ia * dx * dx + ib * (dy * dy + dz * dz)
            B = 2.0 * (ia * ox * dx + ib * (oy * dy + oz * dz))
            C = ia * ox * ox + ib * (oy * oy + oz * oz) - 1.0
            disc = B * B - 4.0 * A * C
            if disc > 0.0 and A > 1e-12:
                th = (-B - ti.sqrt(disc)) / (2.0 * A)
                if th > 1e-6:
                    t_hit = th
    return t_hit


@ti.func
def _body_normal(k: ti.i32, p: ti.template()):
    """天体 k 表面世界点 p 处的外法线（椭球广义）。"""
    rel = p - star_pos_f[k]
    f = star_stretch_f[k]
    r = star_rad_f[k]
    n = rel.normalized()            # 球面法线（轻微拉伸时的近似）
    if f >= 1.02:                   # 椭球广义法线（梯度方向）
        sa = r * f
        sb = r / ti.sqrt(f)
        ax = star_axis_f[k]
        e2, e3 = _axis_basis(ax)
        nx = rel.dot(ax) / (sa * sa)
        ny = rel.dot(e2) / (sb * sb)
        nz = rel.dot(e3) / (sb * sb)
        n = (ax * nx + e2 * ny + e3 * nz).normalized()
    return n


@ti.func
def _occl_frac(t_hit: ti.f32, t_mid: ti.f32, w: ti.f32) -> ti.f32:
    """深度占据 [t_mid-w, t_mid+w] 的发光介质被 t_hit 处不透明面遮挡后的
    剩余可见比例（0=全遮，1=全可见）。

    旧版二值遮挡（t_hit < t_mid - r 即全灭）会在前景星缘处把后星
    辉光/日珥/等离子流硬生生切断 —— 两星靠近时在接触带形成
    “分界线”伪影；改为按介质深度层比例平滑过渡后，后星介质如
    薄纱般覆盖在前星球面上，随深度淡入淡出。
    """
    return min(max((t_hit - (t_mid - w)) / (2.0 * w), 0.0), 1.0)


@ti.func
def _prominence(k: ti.i32, t: ti.f32, bf: ti.f32, d2: ti.f32, r: ti.f32,
                hit_k: ti.i32, t_min: ti.f32, cam: ti.template(),
                rd: ti.template()):
    """临边日珥：贴着星缘外侧的 H-alpha 色等离子环（随时间缓慢演化）。

    只在临边环带（0.9 < ρ < 1.6）内计算；活动度与黑子共用种子
    （sd.y 大 = 磁活动强 = 黑子多且日珥盛），保持表面/临边一致。
    bf: 星心的前向距离；d2: 射线到星心距离²；hit_k/t_min: 遮挡信息。
    """
    col = ti.Vector([0.0, 0.0, 0.0])
    rho2 = d2 / (r * r)
    if rho2 > 0.9 and rho2 < 2.6:
        vis = 1.0
        if hit_k >= 0 and hit_k != k:
            # 环拱纵深 ~PROM_DEPTH·r，按深度层平滑遮挡（hit_k==k 的
            # 盘上像素 t_min >= bf-r 恒成立，不会遮挡自身环拱）
            vis = _occl_frac(t_min, bf, PROM_DEPTH * r)
        if vis > 0.0:
            # 射线上离星心最近的点 -> 临边方向（等离子环的角坐标）
            q = cam + bf * rd
            ldir = (q - star_pos_f[k]).normalized()
            sd = star_seeds[k]
            act = 0.30 + 0.70 * sd.y       # 活动度（与黑子同源）
            lp = _fbm3(ldir * 2.4 + 17.0 * sd
                       + ti.Vector([0.030 * t, 0.0, 0.020 * t]))
            lp = max(lp, 0.0) ** 2.2       # 稀疏的环拱亮斑
            rho = ti.sqrt(rho2)
            prof = ti.exp(-((rho - 1.0) / PROM_SIG) ** 2)
            if lp * prof > 0.004:
                c = _vmix(star_tints[k], ti.Vector([1.0, 0.40, 0.24]), 0.55)
                col += c * (PROM_AMP * act * lp * prof
                            * star_gain_f[k])
    return col


@ti.func
def _contact_glow(t: ti.f32, hit_k: ti.i32, t_min: ti.f32,
                  cam: ti.template(), rd: ti.template()):
    """接触融合辉光：两星近距/接触时填平接触带的辉光凹槽。

    各星日冕内晕是以星心为中心的壳：两星靠近时，接触带只有两侧
    壳的尾部叠加（穿透时还被 self_vis 压零），比星缘更暗 ——
    在融合亮团中呈一条“分界线”把两星切开。本效果为宽幅低峰的
    填充光（接触双星潮汐包层的近似）：横向 σ 远大于旧桥、峰值仅
    补到星缘水平、纵向在两星之间均匀。激活随 d/rs 呈钟形 ——
    盘接触时表面亮度接管、分离较远时暗缝隙属自然分离，唯
    d/rs≈CTR 的“贴近未融合”区间填充最强（实测凹槽最深处）。
    遮挡按包层 σ 纵深平滑。

    近距星对由 CPU 每帧筛选上传（fuse_*，≤MAX_FUSE 对）：N 体时
    O(N²) 逐对扫描对每像素过重，而真正有可见贡献的近距离星对
    通常寥寥无几。
    """
    col = ti.Vector([0.0, 0.0, 0.0])
    for p in range(fuse_n[0]):
        i = fuse_i[p]
        j = fuse_j[p]
        if star_rad_f[i] > 0.01 and star_rad_f[j] > 0.01:
            a = star_pos_f[i]
            b = star_pos_f[j]
            u = b - a
            d = u.norm()
            rs = star_rad_f[i] + star_rad_f[j]
            drs = d / rs
            act = ti.exp(-((drs - FUSE_CTR) / FUSE_WID) ** 2)
            if act > FUSE_CUT:
                un = u / d
                w = cam - a
                wr = rd.dot(w)
                wu = un.dot(w)
                bb = rd.dot(un)
                denom = 1.0 - bb * bb
                # 射线与连线（无穷直线）最近点参数
                tr = (wu * bb - wr) / max(denom, 1e-6)
                s = wu + tr * bb               # 连线参数（世界单位）
                s = max(0.0, min(d, s))
                q = a + un * s                 # 连线一侧最近点
                t2 = rd.dot(q - cam)           # 射线一侧参数
                if t2 > 0.02:
                    dist2 = (cam + rd * t2 - q).norm_sqr()
                    sig = (FUSE_SIG0 + FUSE_SIG1 * act) * rs
                    prof = ti.exp(-dist2 / (2.0 * sig * sig))
                    if prof > 0.004:
                        # 纵向：两星心之间均匀，两端 0.5rs 渐出
                        fade_in = _sstep(0.0, 0.5 * rs, s)
                        fade_out = 1.0 - _sstep(d - 0.5 * rs, d, s)
                        span = fade_in * fade_out
                        if span > 0.0:
                            vis = 1.0
                            if hit_k >= 0:
                                # 包层有 σ 纵深：在星缘处平滑浮现/没入
                                vis = _occl_frac(t_min, t2, sig)
                            if vis > 0.0:
                                # 弥散包层的微弱丝缕
                                fil = 0.86 + 0.28 * _vnoise(
                                    q + 0.10 * t * un)
                                c = _vmix(star_tints[i], star_tints[j], 0.5)
                                c = _vmix(c, ti.Vector([1.0, 1.0, 1.0]),
                                          0.40)
                                g = 0.5 * (star_gain_f[i]
                                           + star_gain_f[j])
                                col += c * (FUSE_AMP * act * act * span
                                            * prof * fil * g * vis)
    return col


@ti.func
def _corona_glow(cam: ti.template(), rd: ti.template(), hit_k: ti.i32,
                 t_min: ti.f32, t: ti.f32, bh_front: ti.f32,
                 px: ti.f32, py: ti.f32):
    """解析日冕辉光 + 临边日珥（沿射线累积；自 pipeline 移入，
    供直射与透镜两条路径共用）。

    bh_front：透镜模式下该射线前方最近致密天体的前向距离 ——
    位于其后的恒星辉光会被黑洞吞没（阴影内不应透出背后的光晕）；
    直射模式传 1e30（不启用）。黑洞自身无日冕，跳过。

    性能：N 体时逐像素 O(N) 的辉光/日珥是最大热点 —— CPU 每帧
    预投影每体屏幕包围盒（scr_bb，覆盖盘∪辉光截断范围），像素
    在盒外直接跳过该体全部计算。截断半径按 gamma 1/2.2 编码后
    不可见标定（见 app._BB_GLOW）—— 按线性域估计会低估暗部台阶
    （1e-3 线性经 gamma ≈ 10/255，曾表现为 NS 光晕的方形轮廓）。
    """
    col = ti.Vector([0.0, 0.0, 0.0])
    for k in range(n_body_f[0]):
        if star_rad_f[k] > 0.01 and star_type_f[k] != TYPE_BH \
                and px >= scr_bb[k, 0] and px <= scr_bb[k, 2] \
                and py >= scr_bb[k, 1] and py <= scr_bb[k, 3]:
            oc = cam - star_pos_f[k]
            bf = -oc.dot(rd)
            if bf > 0.0 and bf < bh_front:
                d2 = oc.dot(oc) - bf * bf    # 射线到星心距离的平方
                r = star_rad_f[k]
                rho2 = d2 / (r * r)
                # 自遮挡：盘内不叠加辉光，临边 0.86r~r 平滑过渡
                self_vis = _sstep(0.86, 1.0, rho2)
                s_in = r * GLOW_IN_SIG
                s_out = r * GLOW_OUT_SIG
                # 深度平滑遮挡：前景不透明面只遮层深超过它的部分
                vis_in = 1.0
                vis_out = 1.0
                if hit_k >= 0 and hit_k != k:
                    vis_in = _occl_frac(t_min, bf, s_in)
                    vis_out = _occl_frac(t_min, bf, s_out)
                # 内晕偏白热（色球/散射），外晕保持星色调（日冕）
                c_in = _vmix(star_tints[k], ti.Vector([1.0, 1.0, 1.0]), 0.45)
                col += c_in * (GLOW_IN_AMP
                               * ti.exp(-d2 / (s_in * s_in) * 3.0)
                               * vis_in * self_vis * star_gain_f[k])
                col += star_tints[k] * (GLOW_OUT_AMP
                                        * ti.exp(-d2 / (s_out * s_out))
                                        * vis_out * self_vis
                                        * star_gain_f[k])
                # 临边日珥（仅环带像素有非零贡献）
                col += _prominence(k, t, bf, d2, r, hit_k, t_min, cam, rd)
    return col


@ti.func
def _weak_deflect(cam: ti.template(), rd: ti.template()):
    """远场解析单次偏折：光线方向朝质量源弯折 α = 2·R_s/b。

    弱场 GR 偏折角 α = 4Gm/(c²b) = 2R_s/b 的精确值；逐个致密天体
    顺序应用（小角叠加）。仅处理撞击参数 > LENS_CUT·R_s 的贡献
    —— 更近的像素由 _photon_march 全程积分，边界处两法偏折量
    相等，无缝衔接。致密天体经 lens_* 紧凑列表遍历（N 体碎片
    化后绝大多数槽位是 MS 碎块，与透镜无关）。
    """
    d = rd
    for q in range(lens_n[0]):
        k = lens_k[q]
        if star_type_f[k] >= TYPE_NS and star_rad_f[k] > 0.01:
            oc = star_pos_f[k] - cam
            proj = oc.dot(d)
            if proj > 0.0:
                bv = oc - d * proj
                b = bv.norm()
                rs = 2.0 * star_mass_f[k] * C_INV_LIGHT2
                if b > LENS_CUT * rs and b > 1e-4:
                    alpha = min(2.0 * rs / b, 0.7)
                    e = bv / b
                    d = (d * ti.cos(alpha) + e * ti.sin(alpha)).normalized()
    return d


@ti.func
def _disk_emission(k: ti.i32, pc: ti.template(), d: ti.template(),
                   t: ti.f32):
    """黑洞吸积盘单次穿越的（发射色, 不透明度）。盘面过黑洞中心，
    法线 ẑ（与轨道面一致）。

    物理：径向亮度 ~ (r_in/r)²（内缘最热最亮）；开普勒轨道速度
    v = sqrt(GM/r) 的相对论束流 —— 接近侧增亮（多普勒因子乘引力
    红移的 2.5 次幂）；颜色按径向温度渐变（内白热 -> 外橙红），
    蓝移侧偏蓝。图案为随开普勒角速度旋转的 fbm —— 差速旋转
    自然剪切成螺旋丝缕。
    """
    m = star_mass_f[k]
    rs = star_rad_f[k]                       # BH 半径即 R_s
    r_in = DISK_IN * rs
    r_out = DISK_OUT * rs
    rel = pc - star_pos_f[k]                 # 盘心跟随黑洞
    rc = ti.sqrt(rel.x * rel.x + rel.y * rel.y)
    u = (rc - r_in) / (r_out - r_in)         # 0 内缘 -> 1 外缘

    # ---- 差速旋转坐标：图案随开普勒角速度转动并被剪切 ----
    omg = DISK_SPIN * t * ti.sqrt(m / max(rc * rc * rc, 1e-6))
    ca = ti.cos(-omg)
    sa = ti.sin(-omg)
    qx = ca * rel.x - sa * rel.y
    qy = sa * rel.x + ca * rel.y
    rn = ti.sqrt(qx * qx + qy * qy)
    inv_r = 1.0 / max(rn, 1e-6)
    rhx = qx * inv_r
    rhy = qy * inv_r
    # 径向高频、切向低频 -> 沿盘向拉长的丝缕
    sx_ = qx * 7.0 - rhy * (rn * 0.9)
    sy_ = qy * 7.0 + rhx * (rn * 0.9)
    n = 0.5 + 0.5 * _fbm3(ti.Vector([sx_, sy_, 23.0 * star_seeds[k].x]))

    # ---- 相对论束流（多普勒 × 引力红移）----
    phi = ti.atan2(rel.y, rel.x)
    vhat = ti.Vector([-ti.sin(phi), ti.cos(phi), 0.0])
    beta = min(ti.sqrt(m / max(rc, 1e-4)) * ti.sqrt(C_INV_LIGHT2), 0.86)
    dopp = ti.sqrt(1.0 - beta * beta) / (1.0 + beta * vhat.dot(d))
    grav = ti.sqrt(max(1.0 - rs / max(rc, 1e-4), 0.04))
    boost = min(max((dopp * grav) ** 2.5, 0.03), 6.0)

    # ---- 颜色：径向温度渐变 + 蓝移色偏 ----
    c = _vmix(ti.Vector([1.0, 0.97, 0.90]), ti.Vector([1.0, 0.64, 0.28]),
              _sstep(0.0, 0.45, u))
    c = _vmix(c, ti.Vector([0.82, 0.26, 0.07]), _sstep(0.45, 1.0, u))
    c = _vmix(c, ti.Vector([0.72, 0.84, 1.0]),
              max(0.0, min(0.45, 0.55 * (dopp - 1.0))))

    e = DISK_AMP * (r_in / rc) ** 2.0 * (0.45 + 0.75 * n) * boost
    e *= _sstep(0.0, 0.06, u) * (1.0 - _sstep(0.80, 1.0, u))  # 内外缘渐隐
    a = min(max(0.28 + 0.45 * n + 0.22 * (1.0 - u), 0.15), 0.92)
    return c * e, a


@ti.func
def _photon_ring(cam: ti.template(), rd: ti.template()):
    """光子环：阴影边缘（临界撞击参数 b ≈ √27/2·R_s ≈ 2.6R_s）
    的细亮环。临界光线绕黑洞多圈后逃逸、背景光被极度放大 ——
    真实物理效应；此处按初始撞击参数的高斯环带作平滑近似
    （避免逐像素多圈混沌采样闪烁），内缘（俘获边界侧）略强。
    """
    col = ti.Vector([0.0, 0.0, 0.0])
    for q in range(lens_n[0]):
        k = lens_k[q]
        if star_type_f[k] >= TYPE_NS and star_rad_f[k] > 0.01:
            rs = 2.0 * star_mass_f[k] * C_INV_LIGHT2
            oc = star_pos_f[k] - cam
            proj = oc.dot(rd)
            if proj > 0.0:
                bv = oc - rd * proj
                b = bv.norm()
                r0 = RING_R * rs
                if ti.abs(b - r0) < 0.5 * r0:
                    g = ti.exp(-((b - r0) / (RING_W * rs)) ** 2)
                    g *= 1.0 + 0.35 * _sstep(r0, r0 * 0.85, b)
                    col += ti.Vector([1.0, 0.88, 0.70]) * (RING_AMP * g)
    return col


@ti.func
def _photon_march(cam: ti.template(), rd: ti.template(), t: ti.f32):
    """近场测地线积分：逐段追踪弯曲光线（引力透镜核心）。

    光子从相机出发沿 rd 以光速前进，按史瓦西零测地线的精确伪力
    偏折（见函数尾注释）后重归一化方向。
    沿弯曲路径依次处理（同段内先近后远）：
      吸积盘平面穿越（BH；半透明前向合成，可多层叠加）；
      恒星表面求交（不透明，被前景盘衰减）；
      视界捕获（r < 1.02·R_s -> 黑；步数耗尽的贴环光线亦视为捕获）。
    返回 (颜色, 透射率, 状态码, 命中星号, 出射方向)；状态码：
    0 逃逸（需补采样背景）、1 命中恒星表面（已着色）、2 捕获/吸收。
    注：Taichi 不支持循环内 early return，用 running 标志位单出口。

    光子动力学：史瓦西零测地线的精确伪力 a = -1.5·R_s·h²/r⁴·r̂
    （h = |w×d| 为角动量）—— 与 Binet 方程 u''+u = 1.5·R_s·u² 严格
    等价，单黑洞下光子球（1.5R_s）、阴影（√27/2·R_s）、弱场偏折
    （α = 2R_s/b）全部精确；多致密天体时取各自伪力之和（无精确
    多体度规，最优可做的线性叠加）。每步重归一化方向（保速 c，
    轨道形状对速度尺度不变，归一化只影响参数化不影响路径）。
    """
    p = cam
    d = rd
    col = ti.Vector([0.0, 0.0, 0.0])
    T = 1.0
    code = 2          # 步数耗尽仍未逃逸 -> 默认捕获
    hit_k = -1
    esc = rd
    running = True
    nl = lens_n[0]                 # 致密天体紧凑列表（march 内 5 处循环共用）
    for _ in range(LENS_STEPS):
        if running:
            # ---- 步长：按最近致密天体距离自适应 ----
            rmin = 1e9
            for qn in range(nl):
                k = lens_k[qn]
                if star_rad_f[k] > 0.01:
                    rmin = min(rmin, (p - star_pos_f[k]).norm())
            h = min(max(0.22 * rmin, LENS_HMIN), LENS_HMAX)
            q = p + d * h

            # ---- 段内吸积盘穿越（BH；盘面过星心 z=star_z，法线 ẑ）----
            for ql in range(nl):
                k = lens_k[ql]
                if star_type_f[k] == TYPE_BH and star_rad_f[k] > 0.01:
                    zk = star_pos_f[k].z
                    if (p.z - zk) * (q.z - zk) < 0.0:
                        s = (p.z - zk) / (p.z - q.z)   # 段内穿越比例
                        pc = p + d * (h * s)
                        rel = pc - star_pos_f[k]
                        rc2 = rel.x * rel.x + rel.y * rel.y
                        rs = star_rad_f[k]
                        if rc2 > DISK_IN * rs * DISK_IN * rs \
                           and rc2 < DISK_OUT * rs * DISK_OUT * rs:
                            e, a = _disk_emission(k, pc, d, t)
                            col += T * e * a
                            T *= 1.0 - a

            # ---- 恒星表面求交（BH 无表面；含潮汐拉伸椭球；全体天体）----
            hk = -1
            t_hit = h + 1e-9
            for k in range(n_body_f[0]):
                if star_type_f[k] < TYPE_BH:
                    th = _body_hit_t(k, p, d)
                    if th > 1e-6 and th < t_hit:
                        t_hit = th
                        hk = k

            if hk >= 0:
                # 命中恒星表面（不透明；前景盘已衰减 T）
                hit_k = hk
                if T > 0.015:
                    ph = p + d * t_hit
                    n = _body_normal(hk, ph)
                    col += T * _star_surface(hk, n, d, t)
                    code = 1
                running = False
            else:
                # ---- 视界捕获 ----
                captured = False
                for ql in range(nl):
                    k = lens_k[ql]
                    if star_type_f[k] == TYPE_BH and star_rad_f[k] > 0.01:
                        rr = (p - star_pos_f[k]).norm()
                        if rr < 1.02 * star_rad_f[k]:
                            captured = True
                if captured:
                    running = False        # code 保持 2
                else:
                    # ---- 推进：史瓦西光子伪力 + 保速 c ----
                    # 逃逸判定：对所有致密天体均在远离（w·d<0）且已
                    # 超出 LENS_ESC·R_s —— 接近中或仍在近场则继续积分
                    acc = ti.Vector([0.0, 0.0, 0.0])
                    escaped = True
                    for ql in range(nl):
                        k = lens_k[ql]
                        if star_rad_f[k] > 0.01:
                            w = star_pos_f[k] - p
                            rr = max(w.norm(), 1e-4)
                            rs = 2.0 * star_mass_f[k] * C_INV_LIGHT2
                            hv = w.cross(d)
                            h2 = hv.norm_sqr()
                            acc += w * (-1.5 * rs * h2 / (rr ** 5))
                            if w.dot(d) >= 0.0 \
                               or rr < LENS_ESC * rs:
                                escaped = False
                    d = (d + h * acc).normalized()
                    p = q
                    if escaped:
                        code = 0
                        esc = d
                        running = False
                    elif T < 0.015:
                        running = False        # 盘几乎不透明 -> 吸收
    return col, T, code, hit_k, esc
