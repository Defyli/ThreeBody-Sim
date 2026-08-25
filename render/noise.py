"""GPU 工具函数（ti.func）：值噪声 / fbm / smoothstep / ACES tone map。

仅依赖 taichi；被 background / star_surface / pipeline 引用。
ti.func 可跨模块被 kernel 内联调用（编译期从本模块全局解析符号）。
"""

import taichi as ti


@ti.func
def _hash_i(n: ti.i32) -> ti.f32:
    """整数哈希 -> [0, 1)（经典值噪声格点哈希）"""
    n = (n << 13) ^ n
    m = n * (n * n * 15731 + 789221) + 1376312589
    return ti.cast(m & 0x7fffffff, ti.f32) * (1.0 / 2147483647.0)


@ti.func
def _vnoise(p: ti.template()) -> ti.f32:
    """3D 值噪声（平滑插值），输出 [-1, 1]"""
    ip = ti.floor(p)
    fp = p - ip
    w = fp * fp * (3.0 - 2.0 * fp)          # 各分量 smoothstep 权重
    i = ti.cast(ip, ti.i32)
    h = i.x + 157 * i.y + 113 * i.z
    n000 = _hash_i(h)
    n100 = _hash_i(h + 1)
    n010 = _hash_i(h + 157)
    n110 = _hash_i(h + 158)
    n001 = _hash_i(h + 113)
    n101 = _hash_i(h + 114)
    n011 = _hash_i(h + 270)
    n111 = _hash_i(h + 271)
    x00 = n000 + (n100 - n000) * w.x
    x10 = n010 + (n110 - n010) * w.y
    x01 = n001 + (n101 - n001) * w.x
    x11 = n011 + (n111 - n011) * w.y
    y0 = x00 + (x10 - x00) * w.y
    y1 = x01 + (x11 - x01) * w.y
    return (y0 + (y1 - y0) * w.z) * 2.0 - 1.0


@ti.func
def _fbm3(p: ti.template()) -> ti.f32:
    """3 阶 fbm（背景星云等用，省性能）"""
    s = 0.0
    a = 0.5
    q = p
    for o in ti.static(range(3)):
        s += a * _vnoise(q)
        q = q * 2.13 + ti.Vector([19.1, 7.7, 11.3])
        a *= 0.5
    return s


@ti.func
def _fbm4(p: ti.template()) -> ti.f32:
    """4 阶 fbm（恒星表面用）"""
    s = 0.0
    a = 0.5
    q = p
    for o in ti.static(range(4)):
        s += a * _vnoise(q)
        q = q * 2.03 + ti.Vector([19.1, 7.7, 11.3])
        a *= 0.5
    return s


@ti.func
def _sstep(e0: ti.f32, e1: ti.f32, x: ti.f32) -> ti.f32:
    t = (x - e0) / (e1 - e0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@ti.func
def _vmix(a: ti.template(), b: ti.template(), s: ti.f32):
    return a + (b - a) * s


@ti.func
def _aces(x: ti.template()):
    """ACES Filmic tone mapping（Narkowicz 2016 拟合，逐分量）"""
    r = ti.Vector([0.0, 0.0, 0.0])
    for ch in ti.static(range(3)):
        v = x[ch]
        v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)
        r[ch] = max(0.0, min(1.0, v))
    return r
