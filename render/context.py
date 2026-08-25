"""Taichi 初始化与全局渲染常量。

本包内所有创建 field 的模块（state / pipeline）都先 import 本模块，
以保证 ti.init 先于任何 field 分配执行。调参集中在这一处。
"""

import taichi as ti

ti.init(arch=ti.gpu)

RES = (1600, 1000)          # 窗口分辨率（像素，须为 4 的倍数）
TAIL_MAX = 4000             # 尾迹环形缓冲容量（点数）
FOV_DEG = 50.0              # 视场角默认值（运行时可经 GUI / 按键调整）

# 恒星色调（RGB；>1 分量经 tone map 呈炽热感）
STAR_TINTS = [
    (0.62, 0.74, 1.00),     # 蓝白
    (1.00, 0.76, 0.42),     # 金橙
    (1.00, 0.42, 0.34),     # 红
]

# ---- 恒星表面着色（逐像素） ----
SURF_BRIGHT = 2.8           # 表面基础亮度（HDR，>1 部分喂给 bloom）
LIMB_U = 0.62               # 临边昏暗系数 u（太阳可见光波段典型值 ~0.6）
GRAN_SCALE = 9.0            # 米粒组织噪声频率（像素级，可比顶点色更细）
SPOT_DARK = 0.85            # 黑子暗度

# ---- 日冕/辉光（解析，沿射线累积） ----
GLOW_IN_SIG = 1.35          # 内晕：贴着球缘的光壳（×恒星半径）
GLOW_IN_AMP = 1.30          # 内晕强度
GLOW_OUT_SIG = 3.6          # 外晕：大范围柔和光晕（×恒星半径）
GLOW_OUT_AMP = 0.14         # 外晕强度

# ---- Bloom（Unreal/CoD 式：亮部提取 + 降采样高斯 + 上采样合成） ----
BLOOM_THR = 0.9             # 亮部阈值（HDR 空间）
BLOOM_SIGMA = 4.0           # 高斯 sigma（1/4 分辨率下，等效全屏 16px）

# ---- 曝光 ----
EXPOSURE_DEF = 1.0
