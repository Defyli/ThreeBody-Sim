"""固定形状的 Taichi field（与窗口分辨率无关，import 时即分配）。

注意：必须先 ti.init 再创建 field —— 本模块通过 import context 保证
执行顺序，请勿调整该 import 的位置。
"""

import taichi as ti

from .context import FOV_DEG, TAIL_MAX

# ---- 每帧动态输入（全局固定形状） ----
cam_pos_f = ti.Vector.field(3, ti.f32, shape=1)
cam_look_f = ti.Vector.field(3, ti.f32, shape=1)
cam_fov_f = ti.field(ti.f32, shape=1)               # 视场角（度，运行时可调）
cam_fov_f[0] = FOV_DEG                              # 默认值（首帧上传前兜底）
star_pos_f = ti.Vector.field(3, ti.f32, shape=3)
star_rad_f = ti.field(ti.f32, shape=3)
star_tints = ti.Vector.field(3, ti.f32, shape=3)
star_seeds = ti.Vector.field(3, ti.f32, shape=3)   # 每星种子：噪声偏移/自转倾角/转速
star_gain_f = ti.field(ti.f32, shape=3)            # 每星亮度增益（按色调亮度归一）
star_mass_f = ti.field(ti.f32, shape=3)            # 质量（透镜：R_s = 2m/c²）
star_type_f = ti.field(ti.i32, shape=3)            # 天体类型（physics.TYPE_*）

# ---- 尾迹（环形展开后上传，前 trail_cnt 个有效） ----
trail_pts = ti.Vector.field(3, ti.f32, shape=(3, TAIL_MAX))
trail_cnt = ti.field(ti.i32, shape=3)
