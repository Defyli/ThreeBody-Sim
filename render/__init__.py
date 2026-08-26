"""Taichi 渲染包。

模块一览：
    context        ti.init + 全局渲染常量（调参集中地）
    noise          GPU 噪声/曲线工具（ti.func）
    background     深空背景着色（银河 / 星云 / 星系 / 星场）
    star_surface   恒星表面着色（米粒 / 黑子 / 临边昏暗……）
    effects        动态特效（临边日珥 / 潮汐等离子桥）
    state          固定形状 field（import 即分配，可安全 from-import）
    pipeline       分辨率相关 field（ensure_fields）+ 全部渲染 kernel

典型用法：
    from render import pipeline
    pipeline.ensure_fields(w, h)
    pipeline.render_scene(t)
    pipeline.splat_trails()
    ...
    canvas.set_image(pipeline.img_tex)

注意：pipeline.img_tex / IMG_W 等会被 ensure_fields 重新绑定，必须
通过模块属性访问，不要 from-import 后长期持有（详见 pipeline 文档）。
"""

from . import (background, context, effects, noise, pipeline,  # noqa: F401
               star_surface, state)
