"""
三体问题 3D 宇宙模拟（Taichi 自绘 HDR 渲染管线）
=================================================

在原 sim.py（matplotlib 动画）的基础上，用 Taichi 实现的实时高画质渲染：

  - 自研 HDR 渲染管线（Texture 直呈，全程 GPU）：
      * 射线-球精确求交，逐像素着色：超米粒 + 域扭曲米粒组织、色球网络亮线、
        黑子（本影/半影丝缕）+ 谱斑、白炽亮斑、临边昏暗（I ~ 1 - u(1-mu)）、
        针状体色球红缘
      * 解析日冕辉光：沿射线到星心距离的指数衰减（内晕贴球缘 + 外层大范围光晕），
        带遮挡判断（被前方天体挡住的辉光自动衰减）
      * Unreal/CoD 式 Bloom：亮部提取 -> 1/4 分辨率可分离高斯 -> 双线性上采样合成
      * ACES Filmic Tone Mapping + gamma 2.2
  - 程序化深空背景：银河（窄亮脊 + 宽盘 + 核球 + 尘埃暗隙 + 恒星颗粒星流）
    + 域扭曲 fbm 发射星云（H-alpha/OIII 双色调 + 丝缕）+ 三个远方旋涡星系
    + 双层黑体色星场（亮星带十字衍射芒）
  - HDR 尾迹：线段光栅化 + 原子加法叠加，渐变渐隐
  - 电影式环绕运镜 / 自由飞行相机（WASD + 鼠标右键）
  - 实时 GUI：暂停、时间倍率、尾迹长度、曝光、辉光强度、初始配置切换（含
    三体精确特解：八字形、拉格朗日等边三角形、欧拉共线、蝴蝶/飞蛾/阴阳
    等 Šuvakov–Dmitrašinović 周期解）、三星质量实时调节（不重置轨迹）、
    镜头 FOV 变焦与环绕推拉

物理内核沿用 RK4 积分（float64、numpy 向量化），保证混沌轨迹精度；
全部渲染在 GPU（Vulkan/Metal）上逐像素完成。

代码结构（按模块拆分）：
    sim3d.py            命令行入口（本文件）：参数解析 / 自检 / 录制
    physics.py          物理内核：三体引力 + RK4 与初始配置（纯 numpy）
    camera.py           相机控制：电影环绕运镜 <-> 自由飞行
    trails.py           尾迹：numpy 环形缓冲 + 上传 GPU
    app.py              应用主体：ThreeBodyUniverse（GUI / 交互 / 主循环 / 自检）
    render/             Taichi 渲染包：
        context.py          ti.init 与全局渲染常量（调参集中地）
        noise.py            GPU 噪声/曲线工具（hash / fbm / smoothstep / ACES）
        background.py       深空背景（银河 / 星云 / 星系 / 星场）
        star_surface.py     恒星表面逐像素着色（米粒 / 黑子 / 临边昏暗……）
        state.py            固定形状 field（相机 / 恒星 / 尾迹）
        pipeline.py         分辨率相关 field + 全部渲染 kernel（管线本体）

运行（在已安装 taichi 的环境，例如 conda 的 lfy 环境）：
    python sim3d.py                              # 交互模式
    python sim3d.py --selftest                   # 离屏渲染若干时刻并保存截图
    python sim3d.py --record shots --frames 300  # 保存帧序列（可合成视频）

快捷键：
    SPACE  暂停/继续        R   重置当前配置
    C      切换相机模式     1-8 切换初始配置（含三体特解）
    M      复位三星质量     ESC 退出
    - / =  镜头变焦（长焦/广角，两种相机模式通用）
    Z / X  推拉镜头（电影模式拉近/拉远，也可用 GUI 滑条）
    自由相机：按住鼠标右键转动视角，WASD 平移，Q/E 升降
    电影相机：按住鼠标左键拖拽转动视角（叠加在自动运镜上）
"""

import argparse

from app import ThreeBodyUniverse, run_selftest
from physics import CONFIG_ORDER


def main():
    parser = argparse.ArgumentParser(description='三体问题 3D 宇宙模拟（Taichi）')
    parser.add_argument('--selftest', action='store_true',
                        help='离屏渲染若干时刻截图并退出')
    parser.add_argument('--record', metavar='DIR', default=None,
                        help='将每帧截图保存到 DIR（用于合成视频）')
    parser.add_argument('--frames', type=int, default=300,
                        help='--record 模式的帧数')
    parser.add_argument('--config', default='pythagorean',
                        choices=CONFIG_ORDER, help='初始配置')
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return
    app = ThreeBodyUniverse(args.config,
                            record_dir=args.record,
                            record_frames=args.frames)
    app.run()


if __name__ == '__main__':
    main()
