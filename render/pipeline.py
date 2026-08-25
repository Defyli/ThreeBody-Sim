"""分辨率相关 field（ensure_fields）+ 全部渲染 kernel。

渲染管线（调用顺序见 app.ThreeBodyUniverse.render）：
    render_scene   背景 + 三恒星精确求交 + 逐像素表面 + 解析日冕辉光
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
from .context import (BLOOM_SIGMA, GLOW_IN_AMP, GLOW_IN_SIG,
                      GLOW_OUT_AMP, GLOW_OUT_SIG)
from .noise import _aces, _vmix
from .star_surface import _star_surface
from .state import (cam_fov_f, cam_look_f, cam_pos_f, star_gain_f,
                    star_pos_f, star_rad_f, star_tints, trail_cnt,
                    trail_pts)

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


@ti.kernel
def render_scene(t: ti.f32):
    """主渲染：背景 + 三恒星精确求交 + 逐像素表面 + 解析日冕辉光 -> img_hdr。"""
    cam, fwd, right, up = _camera_basis()
    tanh = ti.tan(cam_fov_f[0] * 0.008726646)    # tan(fov/2)，运行时可调
    aspect = IMG_W / IMG_H
    for x, y in img_hdr:
        sx = (2.0 * (x + 0.5) / IMG_W - 1.0) * tanh * aspect
        sy = (2.0 * (y + 0.5) / IMG_H - 1.0) * tanh
        rd = (fwd + right * sx + up * sy).normalized()

        col = _bg_color(rd)

        # --- 射线-球求交（rd 已归一化） ---
        t_min = 1e30
        hit_k = -1
        for k in ti.static(range(3)):
            oc = cam - star_pos_f[k]
            b = oc.dot(rd)
            c0 = oc.dot(oc) - star_rad_f[k] * star_rad_f[k]
            disc = b * b - c0
            if disc > 0.0:
                th = -b - ti.sqrt(disc)
                if th > 0.0 and th < t_min:
                    t_min = th
                    hit_k = k

        if hit_k >= 0:
            p = cam + rd * t_min
            n = (p - star_pos_f[hit_k]).normalized()
            col = _star_surface(hit_k, n, rd, t)

        # --- 解析日冕辉光（未命中处也累加；被前方天体遮挡则衰减） ---
        for k in ti.static(range(3)):
            oc = cam - star_pos_f[k]
            b = oc.dot(rd)
            if b > 0.0:
                d2 = oc.dot(oc) - b * b            # 射线到星心距离的平方
                r = star_rad_f[k]
                vis = 1.0
                if hit_k >= 0 and hit_k != k and t_min < b - r:
                    vis = 0.0                      # 辉光中心被前方天体挡住
                s_in = r * GLOW_IN_SIG
                s_out = r * GLOW_OUT_SIG
                # 内晕偏白热（色球/散射），外晕保持星色调（日冕）
                c_in = _vmix(star_tints[k], ti.Vector([1.0, 1.0, 1.0]), 0.45)
                col += c_in * (GLOW_IN_AMP * ti.exp(-d2 / (s_in * s_in) * 3.0)
                               * vis * star_gain_f[k])
                col += star_tints[k] * (GLOW_OUT_AMP * ti.exp(-d2 / (s_out * s_out))
                                        * vis * star_gain_f[k])

        img_hdr[x, y] = col


@ti.kernel
def splat_trails():
    """尾迹：屏幕空间线段光栅化，HDR 原子叠加（越新越亮，渐变渐隐）。

    性能：Taichi 只自动并行最外层 range 循环。若以 k 星循环作外层，
    整个 GPU 仅 3 个线程串行处理上千条线段（曾是主要瓶颈），因此把
    (k, i) 展平成一维 range 统一调度。
    """
    cam, fwd, right, up = _camera_basis()
    tanh = ti.tan(cam_fov_f[0] * 0.008726646)
    aspect = IMG_W / IMG_H
    nseg = trail_cnt[0] - 1
    for s in range(3 * nseg):
        k = s // nseg
        i = s - k * nseg
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
