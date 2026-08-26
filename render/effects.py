"""动态视觉特效（ti.func）：解析日冕辉光的临边日珥 + 潮汐等离子桥。

由 pipeline.render_scene 在辉光阶段调用。两者都只在少数像素上
有非零贡献（日珥局限于临边环带、等离子桥仅在双星近距时激活），
平均每像素开销可忽略。

注意：本模块函数被 pipeline 的 kernel 调用时，模块级全局常量按
编译时绑定内联（与 render 包其他模块同一约束）。
"""

import taichi as ti

from .context import PROM_AMP, PROM_SIG, TIDE_AMP, TIDE_OFF, TIDE_ON, TIDE_SIG
from .noise import _fbm3, _sstep, _vmix, _vnoise
from .state import star_gain_f, star_pos_f, star_rad_f, star_seeds, star_tints


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
        if hit_k >= 0 and t_min < bf - r:
            vis = 0.0                      # 被任何前景星盘挡住
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
def _tidal_bridge(t: ti.f32, hit_k: ti.i32, t_min: ti.f32,
                  cam: ti.template(), rd: ti.template()):
    """潮汐等离子桥：双星近距时沿连线的发光物质流（质量转移近似）。

    两星间距 d < TIDE_ON·(r_i+r_j) 时激活，亮度随接近平滑增强；
    截面高斯（σ = TIDE_SIG·(r_i+r_j)），纵向抛物线渐弱，沿线低频
    噪声调制出丝缕质感；被前景星盘遮挡。
    """
    col = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        for j in ti.static(range(i + 1, 3)):
            a = star_pos_f[i]
            b = star_pos_f[j]
            u = b - a
            d = u.norm()
            rs = star_rad_f[i] + star_rad_f[j]
            act = 1.0 - _sstep(TIDE_OFF * rs, TIDE_ON * rs, d)
            if act > 0.002:
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
                    sig = TIDE_SIG * rs
                    prof = ti.exp(-dist2 / (2.0 * sig * sig))
                    if prof > 0.004:
                        vis = 1.0
                        if hit_k >= 0 and t_min < t2:
                            vis = 0.0          # 前景星盘挡住
                        if vis > 0.0:
                            lt = 4.0 * (s / d) * (1.0 - s / d)  # 纵向渐弱
                            fil = 0.55 + 0.90 * _vnoise(
                                q + 0.12 * t * un)               # 丝缕
                            c = _vmix(star_tints[i], star_tints[j], 0.5)
                            c = _vmix(c, ti.Vector([1.0, 1.0, 1.0]), 0.40)
                            g = 0.5 * (star_gain_f[i] + star_gain_f[j])
                            col += c * (TIDE_AMP * act * act * lt * prof
                                        * fil * g)
    return col
