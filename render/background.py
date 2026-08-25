"""程序化深空背景着色（ti.func）。

银河（窄亮脊 + 宽盘 + 核球 + 尘埃暗隙 + 未解析恒星颗粒星流）
+ 发射星云（IQ 双层域扭曲 fbm + 脊状丝缕，H-alpha/OIII 双色调）
+ 三个远方旋涡星系 + 双层彩色星场（亮星带十字衍射芒）。
"""

import taichi as ti

from .noise import _fbm3, _hash_i, _sstep, _vmix, _vnoise


@ti.func
def _galaxy(rd: ti.template(), c: ti.template(), rad: ti.f32, incl: ti.f32,
            n_arms: ti.f32, tight: ti.f32, phase: ti.f32, amp: ti.f32):
    """远方旋涡星系（billboard）：对数螺旋密度波 + 核球 + 倾斜盘 + 恒星颗粒。

    c: 星系中心方向；rad: 角半径；incl: 盘面倾角（0=正视）；n_arms: 臂数；
    tight: 螺旋松紧；amp: 总亮度。
    """
    col = ti.Vector([0.0, 0.0, 0.0])
    w = rd.dot(c)                       # 朝向权重（只画朝向半球的）
    if w > 0.2:
        a = c.cross(ti.Vector([0.0, 1.0, 0.0]))
        an = a.norm()
        e1 = ti.select(an > 1e-4, a / (an + 1e-9), ti.Vector([1.0, 0.0, 0.0]))
        e2 = c.cross(e1)
        # 切平面投影（近似角坐标），v 轴按倾角压扁成椭圆盘
        u = rd.dot(e1) / w
        v = rd.dot(e2) / (w * max(ti.cos(incl), 0.25))
        rr = ti.sqrt(u * u + v * v) / rad
        if rr < 1.6:
            th = ti.atan2(v, u)
            arm = ti.cos(n_arms * th + tight * ti.log(rr + 0.03) + phase)
            arm = arm * arm                     # 密度波（旋臂亮、臂间暗）
            disk = ti.exp(-rr * rr * 3.0)
            bulge = ti.exp(-rr * rr * 22.0)
            grain = max(_vnoise(rd * 1200.0), 0.0) ** 2.0   # 未解析恒星颗粒
            core_c = ti.Vector([1.0, 0.88, 0.70])   # 老年恒星暖核
            arm_c = ti.Vector([0.55, 0.68, 1.00])   # 年轻蓝臂
            col = (core_c * (1.6 * bulge + 0.22 * disk)
                   + arm_c * (1.05 * disk * arm)) * amp * w
            col *= 0.70 + 0.65 * grain
    return col


@ti.func
def _bg_color(rd: ti.template()):
    """程序化深空背景。

    银河（窄亮脊 + 宽盘 + 核球 + 尘埃暗隙 + 未解析恒星颗粒星流）
    + 发射星云（IQ 双层域扭曲 fbm + 脊状丝缕，H-alpha/OIII 双色调）
    + 三个远方旋涡星系 + 双层彩色星场（亮星带十字衍射芒）。
    """
    # ---------- 银河坐标（axis = 银道面法线，e2 方向为银心） ----------
    axis = ti.Vector([0.34, 0.62, 0.71]).normalized()
    a0 = axis.cross(ti.Vector([1.0, 0.0, 0.0]))
    e1 = a0.normalized()
    e2 = axis.cross(e1)
    zg = rd.dot(axis)
    lat2 = zg * zg
    band_n = ti.exp(-lat2 / 0.0045)     # 窄亮脊（银盘主平面）
    band_w = ti.exp(-lat2 / 0.085)      # 宽盘（晕 + 厚盘）
    cd = rd.dot(e2)
    core = ti.exp(-(1.0 - max(cd, 0.0)) ** 2 / 0.055)   # 核球 ~13°

    # ---------- 尘埃暗隙（沿银道面拉长的 fbm，即"大暗隙"） ----------
    rift = _fbm3(ti.Vector([rd.dot(e1) * 15.0, rd.dot(e2) * 15.0,
                            zg * 60.0]) + ti.Vector([31.7, 11.3, 5.1]))
    rift = _sstep(0.02, 0.42, rift)
    rift_m = rift * (0.30 * band_w + 0.70 * band_n)     # 集中于银道面
    att = 1.0 - 0.78 * rift_m                           # 尘埃乘性遮蔽

    # ---------- 银河漫射光 + 未解析恒星颗粒 ----------
    warm = ti.Vector([1.0, 0.87, 0.68])    # 核球方向老年恒星
    cool = ti.Vector([0.62, 0.72, 1.0])    # 盘面年轻恒星
    col = ti.Vector([0.005, 0.006, 0.013])
    col += warm * (0.020 * band_n * (0.35 + 0.65 * core)
                   + 0.010 * core * band_w)
    col += cool * (0.0085 * band_w)
    haze = max(_vnoise(rd * 300.0), 0.0) ** 3.0         # 星流颗粒
    col += (warm * 0.6 + cool * 0.4) * (0.16 * haze * band_n)
    col += cool * (0.05 * haze * band_w)
    col *= att

    # ---------- 发射星云（双层域扭曲 fbm，IQ 风格） ----------
    p0 = rd * 3.1 + ti.Vector([4.7, 1.3, 8.9])
    q = _fbm3(p0)
    r = _fbm3(p0 + 4.0 * q + ti.Vector([1.7, 9.2, 3.3]))
    f = _fbm3(p0 + 4.0 * r + ti.Vector([8.3, 2.8, 5.6]))
    fil = 1.0 - abs(_fbm3(p0 * 2.3 + 2.5 * r))          # 脊状丝缕
    fil = fil * fil

    weight = band_w * 0.9 + 0.25 * _sstep(0.15, 0.7, f)  # 沿带为主 + 离带补丁
    neb = _sstep(-0.25, 0.75, f) * weight
    neb *= 1.0 + 0.9 * fil
    neb *= 1.0 - 0.5 * rift_m                            # 尘埃部分遮蔽星云

    ha = ti.Vector([1.0, 0.28, 0.32])                    # H-alpha 红
    o3 = ti.Vector([0.28, 0.80, 0.78])                   # OIII 青
    nebc = _vmix(ha, o3, _sstep(-0.2, 0.6, r))
    nebc = _vmix(nebc, ti.Vector([1.0, 0.66, 0.40]), 0.30 * core)
    col += nebc * (0.045 * neb)

    # ---------- 远方星系（大斜旋涡 / 小正视蓝旋涡 / 侧视针状） ----------
    g_att = 1.0 - 0.5 * rift_m
    col += _galaxy(rd, ti.Vector([-0.48, 0.22, 0.85]).normalized(),
                   0.085, 0.95, 2.0, 4.5, 1.7, 0.055) * g_att
    col += _galaxy(rd, ti.Vector([0.77, 0.55, 0.33]).normalized(),
                   0.040, 0.25, 3.0, 5.5, 4.0, 0.030) * g_att
    col += _galaxy(rd, ti.Vector([0.35, -0.75, -0.55]).normalized(),
                   0.100, 1.40, 2.0, 3.0, 0.6, 0.034) * g_att

    # ---------- 星场近层（亮星，黑体色渐变 + 衍射芒） ----------
    K = 150.0
    p = rd * K
    cell = ti.floor(p)
    ic = ti.cast(cell, ti.i32)
    h = _hash_i(ic.x + 157 * ic.y + 113 * ic.z)
    if h < 0.13:
        hb = ti.cast(h * 8191.0, ti.i32)
        u1 = _hash_i(hb + 101)
        u2 = _hash_i(hb + 211)
        u3 = _hash_i(hb + 307)
        u4 = _hash_i(hb + 401)
        # 星心约束在 cell 中央区域，保证完整的圆点
        sp = cell + 0.25 + 0.5 * ti.Vector([u1, u2, u3])
        off = p - sp
        d = off.norm() / K                   # 近似角距离（弧度）

        # 黑体色渐变：蓝白 -> 白 -> 金 -> 橙红
        c = ti.Vector([0.67, 0.78, 1.0])
        c = _vmix(c, ti.Vector([1.0, 0.98, 0.94]), _sstep(0.30, 0.55, u4))
        c = _vmix(c, ti.Vector([1.0, 0.84, 0.62]), _sstep(0.62, 0.85, u4))
        c = _vmix(c, ti.Vector([1.0, 0.62, 0.42]), _sstep(0.88, 0.99, u4))

        bright = 0.14 + 0.60 * _hash_i(hb + 509) ** 3.0
        is_big = _hash_i(hb + 607) < 0.022
        big_gain = ti.select(is_big, 3.2, 1.0)
        sigma = ti.select(is_big, 0.0026, 0.0011)  # 角半径
        lum = bright * big_gain * (1.0 + 1.5 * band_w)
        lum *= 1.0 - 0.55 * rift_m           # 尘埃遮蔽背景恒星
        col += c * (lum * ti.exp(-(d * d) / (sigma * sigma)))
        if is_big:
            # 十字衍射芒（沿银河坐标轴的两条细长高斯）
            aa = off.dot(e1)
            ab = off.dot(e2)
            p1 = (off - e1 * aa).norm()
            p2 = (off - e2 * ab).norm()
            spike = ti.exp(-(p1 * p1) / 0.10) * ti.exp(-ti.abs(aa) / 2.2) \
                  + ti.exp(-(p2 * p2) / 0.10) * ti.exp(-ti.abs(ab) / 2.2)
            col += c * (0.55 * lum * spike)

    # ---------- 星场远层（更小更密更暗，增加纵深） ----------
    K2 = 430.0
    p2v = rd * K2
    c2v = ti.floor(p2v)
    ic2 = ti.cast(c2v, ti.i32)
    h2 = _hash_i(ic2.x + 157 * ic2.y + 113 * ic2.z + 7001)
    if h2 < 0.10:
        hb2 = ti.cast(h2 * 8191.0, ti.i32)
        sp2 = c2v + 0.3 + 0.4 * ti.Vector([_hash_i(hb2 + 311),
                                           _hash_i(hb2 + 419),
                                           _hash_i(hb2 + 523)])
        dd = (p2v - sp2).norm() / K2
        lum2 = 0.10 + 0.30 * _hash_i(hb2 + 719) ** 2.0
        lum2 *= 1.0 + 1.2 * band_w
        col += ti.Vector([0.80, 0.86, 1.0]) \
            * (lum2 * ti.exp(-(dd * dd) / (0.0006 * 0.0006)))
    return col
