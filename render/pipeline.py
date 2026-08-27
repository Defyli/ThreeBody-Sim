"""分辨率相关 field（ensure_fields）+ 全部渲染 kernel。

渲染管线（调用顺序见 app.ThreeBodyUniverse.render）：
    render_scene   背景 + 三恒星精确求交 + 逐像素表面 + 解析日冕辉光
                   + 临边日珥 + 接触融合辉光（effects）
    splat_trails   尾迹线段光栅化（HDR 原子叠加）
    bloom_down     亮部提取 + 4x box 降采样
    bloom_blur_h/v 可分离高斯（1/4 分辨率）
    composite      bloom 双线性上采样 + ACES tone map + gamma
    copy_to_texture 写入呈现纹理 -> canvas.set_image 直呈

实现约束：Taichi kernel 编译时把所在模块的全局（IMG_W / img_hdr 等）
按当时的绑定内联为常量 / field 引用，因此 ensure_fields 重绑定这些
全局之后 kernel 才能看到新对象 —— 这要求本模块的 field 分配与全部
kernel 必须写在同一模块，外部只能以 `render.pipeline.img_tex` 这样的
模块属性方式访问（不可 from-import 后长期持有）。
"""

import taichi as ti

from .background import _bg_color
from .context import (BLOOM_SIGMA, C_INV_LIGHT2, LENS_CUT, MAX_BODIES,
                      MAX_FUSE, MAX_LENS, MAX_PART, PART_GAIN, PART_MIN_PX,
                      TAIL_MAX, TYPE_BH)
from .effects import (_body_hit_t, _body_normal, _contact_glow,
                      _corona_glow, _photon_march, _photon_ring,
                      _weak_deflect)
from .noise import _aces, _sstep, _vmix
from .star_surface import _star_surface
from .state import (SI_AXIS, SI_BB, SI_CAM, SI_FOV, SI_FUSEI, SI_FUSEJ,
                    SI_FUSEN, SI_GAIN, SI_LENSK, SI_LENSN, SI_LOOK,
                    SI_NPART, SI_PAL, SI_PCOL, SI_PPOS, SI_PRAD, SI_POS,
                    SI_STR, SI_TCNT, SI_TPRE, SI_TPTS,
                    cam_fov_f, cam_look_f, cam_pos_f, fuse_i, fuse_j,
                    fuse_n, lens_k, lens_n, n_body_f, n_part, part_alpha,
                    part_col, part_pos, part_rad, scr_bb, stage,
                    star_gain_f, star_mass_f, star_pos_f, star_rad_f,
                    star_stretch_f, star_axis_f, star_tints, star_type_f,
                    trail_cnt, trail_prefix, trail_pts)

# ---- 图像缓冲（按窗口分辨率一次性分配，经 ensure_fields 初始化） ----
IMG_W = 0
IMG_H = 0
img_hdr = None      # HDR 场景（线性空间，可 >1）
img_ldr = None      # tone map 后输出（存进 img_tex 再呈现）
img_tex = None      # 呈现纹理（rgba8）：全程留在 GPU，避免 set_image
                   # 经 CPU 中转再上传的往返开销
bloom_a = None      # 1/4 分辨率 bloom ping
bloom_b = None      # 1/4 分辨率 bloom pong


def ensure_fields(w, h):
    """按窗口分辨率分配图像 field（每个进程只允许一种分辨率）。"""
    global IMG_W, IMG_H, img_hdr, img_ldr, img_tex, bloom_a, bloom_b
    assert w % 4 == 0 and h % 4 == 0, f'分辨率须为 4 的倍数: {(w, h)}'
    if img_hdr is not None:
        assert (IMG_W, IMG_H) == (w, h), \
            f'渲染 field 已按 {(IMG_W, IMG_H)} 分配，不支持中途改变分辨率'
        return
    IMG_W, IMG_H = int(w), int(h)
    img_hdr = ti.Vector.field(3, ti.f32, shape=(IMG_W, IMG_H))
    img_ldr = ti.Vector.field(3, ti.f32, shape=(IMG_W, IMG_H))
    img_tex = ti.Texture(ti.Format.rgba8, (IMG_W, IMG_H))
    bloom_a = ti.Vector.field(3, ti.f32, shape=(IMG_W // 4, IMG_H // 4))
    bloom_b = ti.Vector.field(3, ti.f32, shape=(IMG_W // 4, IMG_H // 4))


@ti.kernel
def scatter_stage():
    """staging 分发：把批量上传的动态输入写入各 field（见 state.SI_*）。

    与 CPU 侧逐 field from_numpy（每次 ~0.22ms 固定开销）相比：一次
    上传 + 本 kernel 仅 ~0.05ms。计数/索引以 f32 承载（≤ 2^24 内
    整数精确，槽号/点数远小于此）；无效槽位的残留值无害 —— 所有
    消费 kernel 都以计数（n_body_f/lens_n/fuse_n/n_part/trail_cnt）
    或半径>0 为上界。
    """
    cam_pos_f[0] = ti.Vector([stage[SI_CAM], stage[SI_CAM + 1],
                             stage[SI_CAM + 2]])
    cam_look_f[0] = ti.Vector([stage[SI_LOOK], stage[SI_LOOK + 1],
                               stage[SI_LOOK + 2]])
    cam_fov_f[0] = stage[SI_FOV]
    lens_n[0] = ti.cast(stage[SI_LENSN], ti.i32)
    fuse_n[0] = ti.cast(stage[SI_FUSEN], ti.i32)
    n_part[0] = ti.cast(stage[SI_NPART], ti.i32)
    for q in ti.static(range(MAX_LENS)):
        lens_k[q] = ti.cast(stage[SI_LENSK + q], ti.i32)
    for q in ti.static(range(MAX_FUSE)):
        fuse_i[q] = ti.cast(stage[SI_FUSEI + q], ti.i32)
        fuse_j[q] = ti.cast(stage[SI_FUSEJ + q], ti.i32)
    for k in range(MAX_BODIES):
        star_gain_f[k] = stage[SI_GAIN + k]
        star_stretch_f[k] = stage[SI_STR + k]
        b = SI_POS + k * 3
        star_pos_f[k] = ti.Vector([stage[b], stage[b + 1], stage[b + 2]])
        b = SI_AXIS + k * 3
        star_axis_f[k] = ti.Vector([stage[b], stage[b + 1], stage[b + 2]])
        for c in ti.static(range(4)):
            scr_bb[k, c] = stage[SI_BB + k * 4 + c]
        trail_cnt[k] = ti.cast(stage[SI_TCNT + k], ti.i32)
        for i in range(TAIL_MAX):
            b = SI_TPTS + (k * TAIL_MAX + i) * 3
            trail_pts[k, i] = ti.Vector([stage[b], stage[b + 1],
                                         stage[b + 2]])
    for k in range(MAX_BODIES + 1):
        trail_prefix[k] = ti.cast(stage[SI_TPRE + k], ti.i32)
    for p in range(MAX_PART):
        part_rad[p] = stage[SI_PRAD + p]
        part_alpha[p] = stage[SI_PAL + p]
        b = SI_PPOS + p * 3
        part_pos[p] = ti.Vector([stage[b], stage[b + 1], stage[b + 2]])
        b = SI_PCOL + p * 3
        part_col[p] = ti.Vector([stage[b], stage[b + 1], stage[b + 2]])


@ti.func
def _camera_basis():
    """由 cam_pos_f/cam_look_f 构建正交相机基 (pos, fwd, right, up)。"""
    cam = cam_pos_f[0]
    look = cam_look_f[0]
    fwd = (look - cam).normalized()
    world_up = ti.Vector([0.0, 1.0, 0.0])
    right = world_up.cross(fwd)
    rn = right.norm()
    right = ti.select(rn > 1e-6, right / (rn + 1e-12), ti.Vector([1.0, 0.0, 0.0]))
    up = fwd.cross(right)
    return cam, fwd, right, up


@ti.func
def _project(p: ti.template(), cam: ti.template(), fwd: ti.template(),
             right: ti.template(), up: ti.template(), tanh: ti.f32,
             aspect: ti.f32):
    """世界点 -> 屏幕像素 (px, py, z)。y 轴向上。z<=0 表示在相机背后。"""
    v = p - cam
    z = v.dot(fwd)
    px = -1.0
    py = -1.0
    if z > 0.02:
        px = (v.dot(right) / (z * tanh * aspect) + 1.0) * 0.5 * IMG_W
        py = (v.dot(up) / (z * tanh) + 1.0) * 0.5 * IMG_H
    return px, py, z


@ti.func
def _ray_stars(cam: ti.template(), rd: ti.template(), px: ti.f32, py: ti.f32):
    """射线-天体求交（rd 已归一化，含潮汐拉伸椭球）
    -> (最近命中距离 t_min, 天体号 hit_k)。

    性能：命中点必落在盘投影内 —— 像素在 CPU 预投影的屏幕包围盒
    （scr_bb）外直接跳过求交，N 体时每像素只测 0-2 个候选体。
    """
    t_min = 1e30
    hit_k = -1
    for k in range(n_body_f[0]):
        if px >= scr_bb[k, 0] and px <= scr_bb[k, 2] \
                and py >= scr_bb[k, 1] and py <= scr_bb[k, 3]:
            th = _body_hit_t(k, cam, rd)
            if th > 0.0 and th < t_min:
                t_min = th
                hit_k = k
    return t_min, hit_k


@ti.kernel
def render_scene(t: ti.f32, lens_on: ti.i32):
    """主渲染：背景 + 恒星求交着色 + 日冕辉光/日珥/融合辉光 + 引力透镜。

    lens_on 由 CPU 侧判定（存在 NS/BH 时为 1）：无致密天体走原直射
    线管线（与历史版本逐位一致）；有则启用透镜：撞击参数 <
    LENS_CUT·R_s 的像素逐段积分弯曲光线（恒星表面/吸积盘/视界都
    在弯曲路径上求交，透镜后的背景沿出射方向采样 —— 背景星场随
    引力弯曲成爱因斯坦环），其余像素用解析弱偏折弯折采样方向。
    两条透镜路径在边界处偏折量相等，无缝衔接；bh_front 用于把
    黑洞后方的恒星辉光从阴影里扣除。致密天体经 lens_* 紧凑列表
    遍历（碎片化 N 体时避免逐像素扫全部槽位）。
    """
    cam, fwd, right, up = _camera_basis()
    tanh = ti.tan(cam_fov_f[0] * 0.008726646)    # tan(fov/2)，运行时可调
    aspect = IMG_W / IMG_H
    for x, y in img_hdr:
        sx = (2.0 * (x + 0.5) / IMG_W - 1.0) * tanh * aspect
        sy = (2.0 * (y + 0.5) / IMG_H - 1.0) * tanh
        rd = (fwd + right * sx + up * sy).normalized()
        px = x + 0.5                          # 像素中心（与 scr_bb 同坐标系）
        py = y + 0.5

        col = ti.Vector([0.0, 0.0, 0.0])
        rdb = rd                       # 实际采样/求交方向（弱偏折后）
        bh_front = 1e30                # 前方最近致密天体的前向距离
        done = False
        if lens_on == 1:
            march = False
            for q in range(lens_n[0]):
                k = lens_k[q]
                oc = star_pos_f[k] - cam
                proj = oc.dot(rd)
                if proj > 0.0:
                    bv = oc - rd * proj
                    b2 = bv.norm()
                    rs = 2.0 * star_mass_f[k] * C_INV_LIGHT2
                    if b2 < LENS_CUT * rs:
                        march = True
                    if b2 < 2.5 * LENS_CUT * rs:
                        bh_front = min(bh_front, proj)
            if march:
                # ---- 近场：测地线积分（表面/盘/视界均在弯曲路径上） ----
                col, T, code, hit_k, esc = _photon_march(cam, rd, t)
                if code == 0:
                    # 逃逸：沿出射方向补背景 + 光子环（均被前景盘衰减）
                    col += T * (_bg_color(esc) + _photon_ring(cam, rd))
                if code != 2:
                    col += _corona_glow(cam, rd, hit_k, 1e30, t, bh_front,
                                        px, py)
                    col += _contact_glow(t, hit_k, 1e30, cam, rd)
                else:
                    # 捕获：阴影纯黑（前方恒星辉光仍可见）
                    col += _corona_glow(cam, rd, -1, 1e30, t, bh_front,
                                        px, py)
                done = True
            else:
                # ---- 远场：解析弱偏折（背景/星像/辉光沿弯折方向） ----
                rdb = _weak_deflect(cam, rd)
        if not done:
            # ---- 直射管线（无致密天体时与历史版本逐位一致） ----
            col = _bg_color(rdb)
            t_min, hit_k = _ray_stars(cam, rdb, px, py)
            if hit_k >= 0:
                p = cam + rdb * t_min
                n = _body_normal(hit_k, p)      # 潮汐拉伸时为椭球法线
                col = _star_surface(hit_k, n, rdb, t)
            col += _corona_glow(cam, rdb, hit_k, t_min, t, bh_front,
                                px, py)
            col += _contact_glow(t, hit_k, t_min, cam, rdb)

        img_hdr[x, y] = col


@ti.kernel
def splat_trails():
    """尾迹：屏幕空间线段光栅化，HDR 原子叠加（越新越亮，渐变渐隐）。

    性能：Taichi 只自动并行最外层 range 循环。星数不固定（潮汐
    碎片各有尾迹），把 (k, i) 按每星段数前缀和（trail_prefix，
    CPU 侧 upload 时算好）展平成一维 range 统一调度；每段内回溯
    所属星号与段号（静态展开的小扫描）。每星段数独立（并合熄灭
    的星计数为 0，自动不参与绘制）。
    """
    cam, fwd, right, up = _camera_basis()
    tanh = ti.tan(cam_fov_f[0] * 0.008726646)
    aspect = IMG_W / IMG_H
    total = trail_prefix[MAX_BODIES]
    nl = lens_n[0]                       # 致密天体紧凑列表（阴影遮蔽）
    for s in range(total):
        # ---- 回溯所属星号 k 与段号 i（前缀和静态扫描） ----
        k = 0
        i = 0
        acc = 0
        for kk in ti.static(range(MAX_BODIES)):
            c = max(trail_cnt[kk] - 1, 0)
            if s >= acc and s < acc + c:
                k = kk
                i = s - acc
            acc += c
        nseg = max(trail_cnt[k] - 1, 0)
        tint = star_tints[k]
        p0 = trail_pts[k, i]
        p1 = trail_pts[k, i + 1]
        fade = (i + 0.5) / nseg
        fade = fade * fade                     # 二次渐隐
        x0, y0, z0 = _project(p0, cam, fwd, right, up, tanh, aspect)
        x1, y1, z1 = _project(p1, cam, fwd, right, up, tanh, aspect)
        if z0 > 0.02 and z1 > 0.02:
            col = tint * (1.1 * fade + 0.02)
            # 线段包围盒
            pad = 2.5
            xmin = int(max(0.0, ti.floor(min(x0, x1) - pad)))
            xmax = int(min(IMG_W - 1.0, ti.ceil(max(x0, x1) + pad)))
            ymin = int(max(0.0, ti.floor(min(y0, y1) - pad)))
            ymax = int(min(IMG_H - 1.0, ti.ceil(max(y0, y1) + pad)))
            dx = x1 - x0
            dy = y1 - y0
            seg2 = dx * dx + dy * dy
            for u in range(xmin, xmax + 1):
                for v in range(ymin, ymax + 1):
                    # 像素中心到线段的距离
                    d2 = 0.0
                    if seg2 > 1e-9:
                        sp = ((u - x0) * dx + (v - y0) * dy) / seg2
                        sp = max(0.0, min(1.0, sp))
                        qx = x0 + sp * dx - u
                        qy = y0 + sp * dy - v
                        d2 = qx * qx + qy * qy
                    else:
                        d2 = (u - x0) ** 2 + (v - y0) ** 2
                    w = ti.exp(-d2 / (2.0 * 1.3 * 1.3))
                    if w > 0.004:
                        # 黑洞阴影遮罩：尾迹是屏幕空间直线绘制，落入
                        # 阴影（且位于黑洞后方）的段应被吞没，否则会
                        # 漂在纯黑阴影上。阴影角半径 ≈ 2.6·R_s/距离。
                        zmid = 0.5 * (z0 + z1)
                        for qb in range(nl):
                            kb = lens_k[qb]
                            if star_type_f[kb] == TYPE_BH \
                                    and star_rad_f[kb] > 0.01:
                                bx, by, bz = _project(
                                    star_pos_f[kb], cam, fwd, right, up,
                                    tanh, aspect)
                                if bz > 0.02 and zmid > bz:
                                    rpx = (2.598 * star_rad_f[kb] / bz) \
                                        / tanh * (0.5 * IMG_H)
                                    dd = ti.sqrt((u - bx) ** 2
                                                 + (v - by) ** 2)
                                    w *= _sstep(0.92 * rpx, 1.30 * rpx, dd)
                        if w > 0.004:
                            ti.atomic_add(img_hdr[u, v], col * w)


@ti.kernel
def splat_particles():
    """事件粒子：屏幕空间高斯点精灵（HDR 原子叠加，喂 bloom）。

    粒子由 CPU 侧环形缓冲维护（潮汐撕裂喷发 / 碎片蒸发尾 / 并合
    溅射），每帧紧凑上传前 n_part 个有效粒子。世界半径按透视换算
    像素半径；黑洞阴影遮蔽按粒子中心近似（粒子远小于阴影尺度，
    入影即吞没），在像素扫描前完成（每粒子 O(lens_n) 而非每像素）。
    """
    cam, fwd, right, up = _camera_basis()
    tanh = ti.tan(cam_fov_f[0] * 0.008726646)
    aspect = IMG_W / IMG_H
    for p in range(n_part[0]):
        a = part_alpha[p]
        if a > 0.004:
            px, py, pz = _project(part_pos[p], cam, fwd, right, up,
                                  tanh, aspect)
            if pz > 0.02:
                # ---- 黑洞阴影遮蔽（粒子级近似） ----
                for qb in range(lens_n[0]):
                    kb = lens_k[qb]
                    if star_type_f[kb] == TYPE_BH \
                            and star_rad_f[kb] > 0.01:
                        bx, by, bz = _project(star_pos_f[kb], cam, fwd,
                                              right, up, tanh, aspect)
                        if bz > 0.02 and pz > bz:
                            rpx = (2.598 * star_rad_f[kb] / bz) \
                                / tanh * (0.5 * IMG_H)
                            dd = ti.sqrt((px - bx) ** 2 + (py - by) ** 2)
                            a *= _sstep(0.92 * rpx, 1.30 * rpx, dd)
                if a > 0.004:
                    rpx = part_rad[p] / (pz * tanh) * (0.5 * IMG_H)
                    rpx = max(rpx, PART_MIN_PX)
                    col = part_col[p] * (PART_GAIN * a)
                    pad = 2.5 * rpx + 1.5
                    xmin = int(max(0.0, ti.floor(px - pad)))
                    xmax = int(min(IMG_W - 1.0, ti.ceil(px + pad)))
                    ymin = int(max(0.0, ti.floor(py - pad)))
                    ymax = int(min(IMG_H - 1.0, ti.ceil(py + pad)))
                    for u in range(xmin, xmax + 1):
                        for v in range(ymin, ymax + 1):
                            d2 = (u - px) ** 2 + (v - py) ** 2
                            w = ti.exp(-d2 / (2.0 * rpx * rpx))
                            if w > 0.01:
                                ti.atomic_add(img_hdr[u, v], col * w)


@ti.kernel
def bloom_down(thr: ti.f32):
    """亮部提取 + 4x box 降采样到 1/4 分辨率。"""
    for u, v in bloom_a:
        acc = ti.Vector([0.0, 0.0, 0.0])
        for di, dj in ti.ndrange(4, 4):
            acc += img_hdr[u * 4 + di, v * 4 + dj]
        c = acc * 0.0625
        for ch in ti.static(range(3)):
            bloom_a[u, v][ch] = max(c[ch] - thr, 0.0) / (1.0 - thr * 0.5)


@ti.kernel
def bloom_blur_h():
    """水平高斯（sigma = BLOOM_SIGMA，1/4 分辨率）。"""
    bw = IMG_W // 4
    r = 10
    for u, v in bloom_b:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for o in range(-r, r + 1):
            w = ti.exp(-(o * o) / (2.0 * BLOOM_SIGMA * BLOOM_SIGMA))
            uu = min(max(u + o, 0), bw - 1)
            acc += bloom_a[uu, v] * w
            wsum += w
        bloom_b[u, v] = acc / wsum


@ti.kernel
def bloom_blur_v():
    """垂直高斯（sigma = BLOOM_SIGMA，1/4 分辨率）。"""
    bh = IMG_H // 4
    r = 10
    for u, v in bloom_a:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for o in range(-r, r + 1):
            w = ti.exp(-(o * o) / (2.0 * BLOOM_SIGMA * BLOOM_SIGMA))
            vv = min(max(v + o, 0), bh - 1)
            acc += bloom_b[u, vv] * w
            wsum += w
        bloom_a[u, v] = acc / wsum


@ti.kernel
def composite(exposure: ti.f32, bloom_str: ti.f32):
    """合成 bloom（双线性上采样）+ ACES tone map + gamma -> img_ldr。"""
    bw = IMG_W // 4
    bh = IMG_H // 4
    for x, y in img_ldr:
        # 双线性上采样 1/4 分辨率 bloom
        fx = (x + 0.5) * 0.25 - 0.5
        fy = (y + 0.5) * 0.25 - 0.5
        u0 = int(min(max(fx, 0.0), bw - 1.0))
        v0 = int(min(max(fy, 0.0), bh - 1.0))
        u1 = min(u0 + 1, bw - 1)
        v1 = min(v0 + 1, bh - 1)
        tx = min(max(fx - u0, 0.0), 1.0)
        ty = min(max(fy - v0, 0.0), 1.0)
        bl = _vmix(_vmix(bloom_a[u0, v0], bloom_a[u1, v0], tx),
                   _vmix(bloom_a[u0, v1], bloom_a[u1, v1], tx), ty)

        c = (img_hdr[x, y] + bl * bloom_str) * exposure
        c = _aces(c)
        for ch in ti.static(range(3)):
            img_ldr[x, y][ch] = ti.pow(c[ch], 1.0 / 2.2)   # gamma


@ti.kernel
def copy_to_texture(tex: ti.types.rw_texture(num_dimensions=2,
                                             fmt=ti.Format.rgba8)):
    """把 LDR 结果写入呈现纹理：全程留在 GPU，避免 set_image(field)
    经 CPU 中转再回传的往返开销（坐标方向与 field 路径一致，已验证）。"""
    for x, y in img_ldr:
        c = img_ldr[x, y]
        tex.store(ti.Vector([x, y]), ti.Vector([c.x, c.y, c.z, 1.0]))
