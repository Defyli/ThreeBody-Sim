"""固定形状的 Taichi field（与窗口分辨率无关，import 时即分配）。

注意：必须先 ti.init 再创建 field —— 本模块通过 import context 保证
执行顺序，请勿调整该 import 的位置。

天体相关 field 统一按 MAX_BODIES 槽位预分配（潮汐瓦解可动态增生
碎片至该上限）；n_body_f[0] 为当前有效槽位数（含已熄灭的死星 ——
其半径为 0，kernel 内自动跳过），kernel 用它作运行时循环上界。

每帧动态输入经单一 staging 缓冲（stage）批量上传：Metal 上每次
from_numpy 有 ~0.22ms 固定开销，逐 field 直传 ~20 次要 ~4.6ms；
打包后一次上传 + GPU 端 scatter kernel（pipeline.scatter_stage）
分发到各 field，总开销 <1ms。布局偏移（SI_*）在此集中定义，
CPU 打包（app.render）与 GPU 分发（pipeline）共用。
"""

import taichi as ti

from .context import (FOV_DEG, MAX_BODIES, MAX_FUSE, MAX_LENS, MAX_PART,
                      TAIL_MAX)

# ---- 每帧动态输入（全局固定形状） ----
cam_pos_f = ti.Vector.field(3, ti.f32, shape=1)
cam_look_f = ti.Vector.field(3, ti.f32, shape=1)
cam_fov_f = ti.field(ti.f32, shape=1)               # 视场角（度，运行时可调）
cam_fov_f[0] = FOV_DEG                              # 默认值（首帧上传前兜底）
n_body_f = ti.field(ti.i32, shape=1)                # 有效天体槽位数（运行时循环上界）
star_pos_f = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)
star_rad_f = ti.field(ti.f32, shape=MAX_BODIES)
star_tints = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)
star_seeds = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)   # 每星种子：噪声偏移/自转倾角/转速
star_gain_f = ti.field(ti.f32, shape=MAX_BODIES)    # 每星亮度增益（按色调亮度归一）
star_mass_f = ti.field(ti.f32, shape=MAX_BODIES)    # 质量（透镜：R_s = 2m/c²）
star_type_f = ti.field(ti.i32, shape=MAX_BODIES)    # 天体类型（physics.TYPE_*）
star_stretch_f = ti.field(ti.f32, shape=MAX_BODIES)  # 潮汐拉伸因子（1=球；横向按体积守恒压缩）
star_axis_f = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)  # 潮汐轴（单位向量，指向主导扰动体）

# ---- 接触融合辉光的近距离星对（CPU 侧每帧筛选上传，kernel O(pairs)） ----
fuse_n = ti.field(ti.i32, shape=1)
fuse_i = ti.field(ti.i32, shape=MAX_FUSE)
fuse_j = ti.field(ti.i32, shape=MAX_FUSE)

# ---- 致密天体（NS/BH）compact 列表（CPU 每帧上传；透镜/阴影循环） ----
# N 体碎片化后绝大多数是 MS 碎块，逐像素遍历全部槽位浪费 —— 透镜
# 相关 kernel 只需扫这张短表（典型 0-2 个成员）。
lens_n = ti.field(ti.i32, shape=1)
lens_k = ti.field(ti.i32, shape=MAX_LENS)

# ---- 每体屏幕包围盒（CPU 投影上传；kernel 逐体早退） ----
# (x0, y0, x1, y1) 像素矩形，覆盖盘投影 ∪ 辉光截断范围；空盒
# （x0 > x1）表示该体对当前像素群无可见贡献，直接跳过。
scr_bb = ti.field(ti.f32, shape=(MAX_BODIES, 4))

# ---- 事件粒子（CPU 侧环形缓冲打包，有效粒子紧凑上传） ----
part_pos = ti.Vector.field(3, ti.f32, shape=MAX_PART)
part_col = ti.Vector.field(3, ti.f32, shape=MAX_PART)
part_rad = ti.field(ti.f32, shape=MAX_PART)     # 世界半径（屏上换算像素）
part_alpha = ti.field(ti.f32, shape=MAX_PART)   # 亮度包络（出生渐入 + 老化渐隐）
n_part = ti.field(ti.i32, shape=1)

# ---- 尾迹（环形展开后上传，前 trail_cnt 个有效；前缀和供 splat 展平调度） ----
trail_pts = ti.Vector.field(3, ti.f32, shape=(MAX_BODIES, TAIL_MAX))
trail_cnt = ti.field(ti.i32, shape=MAX_BODIES)
trail_prefix = ti.field(ti.i32, shape=MAX_BODIES + 1)   # 每星段数前缀和（展平一维调度用）

# ---- 每帧动态输入的批量上传通道（staging；布局见模块 docstring） ----
SI_CAM = 0                                             # 相机位置 (3)
SI_LOOK = 3                                            # 注视点 (3)
SI_FOV = 6                                             # 视场角 (1)
SI_LENSN = 7                                           # 致密天体数 (1)
SI_FUSEN = 8                                           # 近距星对数 (1)
SI_NPART = 9                                           # 有效粒子数 (1)
SI_LENSK = 10                                          # 致密天体槽位表 (MAX_LENS)
SI_FUSEI = SI_LENSK + MAX_LENS                         # 星对 i 表 (MAX_FUSE)
SI_FUSEJ = SI_FUSEI + MAX_FUSE                         # 星对 j 表 (MAX_FUSE)
SI_GAIN = SI_FUSEJ + MAX_FUSE                          # 每星增益（含闪光） (MAX_BODIES)
SI_STR = SI_GAIN + MAX_BODIES                          # 潮汐拉伸因子 (MAX_BODIES)
SI_POS = SI_STR + MAX_BODIES                           # 位置展平 (3*MAX_BODIES)
SI_AXIS = SI_POS + 3 * MAX_BODIES                      # 潮汐轴展平 (3*MAX_BODIES)
SI_BB = SI_AXIS + 3 * MAX_BODIES                       # 屏幕包围盒展平 (4*MAX_BODIES)
SI_TCNT = SI_BB + 4 * MAX_BODIES                       # 尾迹点数 (MAX_BODIES)
SI_TPRE = SI_TCNT + MAX_BODIES                         # 尾迹前缀和 (MAX_BODIES+1)
SI_PRAD = SI_TPRE + MAX_BODIES + 1                     # 粒子半径 (MAX_PART)
SI_PAL = SI_PRAD + MAX_PART                            # 粒子亮度包络 (MAX_PART)
SI_PPOS = SI_PAL + MAX_PART                            # 粒子位置展平 (3*MAX_PART)
SI_PCOL = SI_PPOS + 3 * MAX_PART                       # 粒子颜色展平 (3*MAX_PART)
SI_TPTS = SI_PCOL + 3 * MAX_PART                       # 尾迹点展平 (3*MAX_BODIES*TAIL_MAX)
STAGE_N = SI_TPTS + 3 * MAX_BODIES * TAIL_MAX
stage = ti.field(ti.f32, shape=STAGE_N)
