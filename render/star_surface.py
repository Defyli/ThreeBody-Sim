"""恒星表面逐像素着色（ti.func）。

层次：慢漂移 + 超米粒（大对流胞）+ 域扭曲米粒组织 + 色球网络亮线
+ 黑子（本影 / 半影丝缕）+ 谱斑（活动区增亮）+ 白炽
+ 临边昏暗 + 色球红缘（针状体噪声调制）。
"""

import taichi as ti

from .context import GRAN_SCALE, LIMB_U, SPOT_DARK, SURF_BRIGHT
from .noise import _fbm3, _fbm4, _sstep, _vmix, _vnoise
from .state import star_gain_f, star_seeds, star_tints


@ti.func
def _star_surface(k: ti.i32, n: ti.template(), rd: ti.template(),
                  t: ti.f32):
    """恒星表面逐像素着色。

    n: 命中点世界法线；rd: 射线方向（指向外）。
    """
    tint = star_tints[k]
    sd = star_seeds[k]
    deep = tint * tint                            # 深色调（更饱和的本色）

    # --- 自转（只旋转噪声采样域，几何不变） ---
    tilt = sd.y * 1.4
    ct = ti.cos(tilt)
    st = ti.sin(tilt)
    y1 = ct * n.y - st * n.z
    z1 = st * n.y + ct * n.z
    ang = 6.2831853 * sd.x + 0.06 * t * (0.5 + sd.z)
    ca = ti.cos(ang)
    sa = ti.sin(ang)
    nrm = ti.Vector([ca * n.x + sa * z1, y1, -sa * n.x + ca * z1])

    # --- 对流：慢漂移 + 超米粒 + 域扭曲米粒（每星粒度略有差异） ---
    tt = 0.10 * t
    gs = GRAN_SCALE * (0.75 + 0.5 * sd.z)
    slow = _fbm4(nrm * 1.9 + 5.0 * sd
                 + ti.Vector([0.05 * tt, -0.04 * tt, 0.03 * tt]))
    sup = _fbm4(nrm * 3.6 + 9.0 * sd
                + ti.Vector([-0.03 * tt, 0.05 * tt, 0.04 * tt]))
    wv = _fbm4(nrm * 2.6 + 7.0 * sd
               + ti.Vector([0.5 * tt, 0.4 * tt, 0.45 * tt]))
    gran = _fbm4(nrm * gs
                 + ti.Vector([wv * 0.8 + tt,
                              wv * 0.8 - 0.7 * tt,
                              wv * 0.8 + 0.35 * tt]))
    # 色球网络：米粒间隙的亮脊（脊状噪声，磁网络）
    net = 1.0 - abs(_fbm4(nrm * (gs * 0.5) + 3.0 * sd + wv * 0.6))
    net = net * net * net
    m = max(0.0, min(1.0, 0.5 + 0.24 * slow + 0.18 * sup + 0.40 * gran))

    # --- 黑子活动区（低频磁场场；种子决定该星的活动度） ---
    # fbm3 实测 σ≈0.22：本影 ~p98.5（约 1% 盘面）、半影环 ~p96
    sf = _fbm3(nrm * 2.4 + 13.0 * sd)
    th = 0.32 - 0.16 * sd.y
    full = _sstep(th + 0.10, th + 0.18, sf)        # 黑子整体（含半影）
    umbra = _sstep(th + 0.18, th + 0.26, sf)       # 本影核心
    pen = full - umbra                             # 半影环
    plage = _sstep(th - 0.06, th + 0.04, sf) * (1.0 - full)   # 谱斑环
    pfil = 0.5 + 0.5 * _vnoise(nrm * 36.0 + 21.0 * sd)  # 半影丝缕

    # --- 颜色映射：暗处深饱和 -> 星色 -> 白炽（HDR>1 喂给 bloom） ---
    c = tint * (0.26 + 0.95 * m)
    c = _vmix(c, deep * (0.34 + 0.50 * m), (1.0 - m) * 0.40)
    hot = _sstep(0.66, 0.95, m)
    c += ti.Vector([1.0, 0.99, 0.94]) * (1.1 * hot * hot)
    # 色球网络亮线 + 谱斑增亮
    c += _vmix(tint, ti.Vector([1.0, 1.0, 1.0]), 0.5) \
        * (0.085 * net * (0.4 + 0.6 * m))
    c += tint * (0.20 * plage)
    # 黑子：半影（丝缕条纹调制）-> 本影（深暗）
    c = _vmix(c, c * (0.62 + 0.26 * pfil), pen * SPOT_DARK)
    spotc = deep * 0.10 + ti.Vector([0.03, 0.005, 0.0])
    c = _vmix(c, spotc, umbra * SPOT_DARK)

    # --- 临边昏暗（物理式 I/I0 ~ 1 - u(1-mu)）+ 色球红缘 ---
    mu = max(0.0, -rd.dot(n))                     # 视线角余弦（精确）
    limb = 1.0 - LIMB_U * (1.0 - mu)
    c *= limb
    edge = 1.0 - mu
    c = _vmix(c, deep * (0.30 + 0.55 * m), 0.42 * edge * edge)
    spic = 0.5 + 0.5 * _vnoise(nrm * 42.0 + 33.0 * sd)   # 针状体
    rim = _vmix(tint, ti.Vector([1.0, 0.50, 0.32]), 0.40)
    c += rim * (0.13 * edge ** 4 * (0.45 + 0.9 * spic))

    return c * SURF_BRIGHT * star_gain_f[k]
