"""
三体问题 3D 宇宙模拟（Taichi 自绘 HDR 渲染管线）
=================================================

在原 sim.py（matplotlib 动画）的基础上，用 Taichi 实现的实时高画质渲染：

  - 自研 HDR 渲染管线（Texture 直呈，全程 GPU）：
      * 射线-球精确求交，逐像素着色：超米粒 + 域扭曲米粒组织、色球网络亮线、
        黑子（本影/半影丝缕）+ 谱斑、白炽亮斑、临边昏暗（I ~ 1 - u(1-mu)）、
        针状体色球红缘
      * 解析日冕辉光：沿射线到星心距离的指数衰减（内晕贴球缘 + 外层大范围光晕），
        带遮挡判断（被前方天体挡住的辉光自动衰减）与盘内自遮挡
      * 临边日珥：星缘外侧的 H-alpha 色等离子环，随时间演化，
        活动度与黑子同源（磁活动强的星黑子多且日珥盛）
      * 接触融合辉光：双星贴近时填平接触带的宽幅低峰填充光
        （接触双星潮汐包层近似），随 d/rs 钟形激活、带丝缕质感与遮挡
      * 引力透镜（中子星 + 黑洞）：光子按粒子以光速沿射线积分，受
        2×牛顿偏折（弱场 GR 精确 α = 4Gm/c²b）—— 黑洞阴影、光子环、
        吸积盘（开普勒差速旋转丝缕 + 相对论多普勒束流：接近侧增亮
        偏蓝、引力红移暗化内缘）、背景星场弯曲成爱因斯坦环；远场
        用解析单次偏折无缝衔接，无致密天体时走原直射管线
      * Unreal/CoD 式 Bloom：亮部提取 -> 1/4 分辨率可分离高斯 -> 双线性上采样合成
      * ACES Filmic Tone Mapping + gamma 2.2
  - 密度驱动的天体类型：主序星 / 白矮星 / 中子星 / 黑洞（GUI 密度
    滑条实时切换；半径按等密度球 / 史瓦西半径计算，致密天体并合
    不降低致密性 —— 黑洞吞并任何天体仍是黑洞）
  - 潮汐物理（变形 + 瓦解，随碰撞开关）：天体在近距时沿潮汐轴
    拉伸成椭球（平衡潮线性项 + 近瓦解非线性陡增，撕裂前可拉长
    ~2.3 倍）；越过洛希极限 d < q·r·(M/m)^{1/3}（q=2.2，近流体）
    即撕裂为真实粒子碎片流（沿潮汐轴排布 + 开普勒剪切 + 解绑
    膨胀：近侧加速坠入、远侧减速逃逸，一半回落吸积、一半甩出
    —— 真实 TDE 图景）；碎片有出生保护期免互撞/免再撕（防级联），
    之后恢复碰撞（回落吸积成团）；等质量流体星总在接触前互相
    撕裂（等密度下洛希半径 = 2.2×对方半径 > 接触距离）
  - 后牛顿动力学（1PN + 2.5PN，始终启用）：轨道进动与自能修正
    （EIH 两体精确项逐对叠加）+ 引力波辐射反作用（能量平衡拖曳，
    圆轨严格 Peters 旋近）—— 致密双星会缓慢旋近并合；周期特解
    （八字形等牛顿精确解）会缓慢退相位并微幅旋近（该宇宙 c²=30
    下星体本就相对论性，属自洽物理而非 bug）
  - 程序化深空背景：银河（窄亮脊 + 宽盘 + 核球 + 尘埃暗隙 + 恒星颗粒星流）
    + 域扭曲 fbm 发射星云（H-alpha/OIII 双色调 + 丝缕）+ 三个远方旋涡星系
    + 双层黑体色星场（亮星带十字衍射芒）
  - HDR 尾迹：线段光栅化 + 原子加法叠加，渐变渐隐
  - 电影式环绕运镜 / 自由飞行相机（WASD + 鼠标右键）
  - 实时 GUI：暂停、时间倍率、尾迹长度、曝光、辉光强度、初始配置切换（含
    三体精确特解：八字形、拉格朗日等边三角形、欧拉共线、蝴蝶/飞蛾/阴阳
    等 Šuvakov–Dmitrašinović 周期解、黑洞+双星）、三星质量与密度实时
    调节（不重置轨迹）、镜头 FOV 变焦与环绕推拉

物理内核沿用 RK4 积分（float64、numpy 向量化，全状态四阶；动力学
= 牛顿 + 1PN + 2.5PN，详见 physics.py 文档），保证混沌轨迹精度；
物理核为变长 N 体（初始 3 体，潮汐瓦解可增生碎片至 28 体）：表面
接触即动量守恒并合（质量相加、色调按质量混合、半径按新质量重算、
并合闪光 + 死星尾迹渐隐），越过洛希极限则撕裂为碎片流（撕裂闪光 +
碎片各自拖尾迹，回落吸积经碰撞并合成团），碰撞/瓦解可在 GUI 开关；
周期特解（八字形/拉格朗日/欧拉/蝴蝶/飞蛾/阴阳）默认关闭碰撞以
保持点质量编舞的完整性；含近碰撞的特解（蝴蝶/飞蛾/阴阳）
采用自适应子步（近距按引力时标 κ·d^1.5 自动加密，能量漂移比固定
细步长低 3-4 个数量级且快约 4.5 倍）；全部渲染在 GPU（Vulkan/Metal）
上逐像素完成。

代码结构（按模块拆分）：
    sim3d.py            命令行入口（本文件）：参数解析 / 自检 / 录制
    physics.py          物理内核：N 体引力 + RK4 与初始配置（纯 numpy，
                        动力学 = 牛顿 + 1PN + 2.5PN 后牛顿修正 +
                        潮汐变形/洛希瓦解）
    camera.py           相机控制：电影环绕运镜 <-> 自由飞行
    trails.py           尾迹：numpy 环形缓冲 + 上传 GPU
    app.py              应用主体：ThreeBodyUniverse（GUI / 交互 / 主循环 / 自检）
    render/             Taichi 渲染包：
        context.py          ti.init 与全局渲染常量（调参集中地）
        noise.py            GPU 噪声/曲线工具（hash / fbm / smoothstep / ACES）
        background.py       深空背景（银河 / 星云 / 星系 / 星场）
        star_surface.py     恒星表面逐像素着色（米粒 / 黑子 / 临边昏暗……）
        effects.py          动态特效（日冕辉光 / 日珥 / 融合辉光 / 引力透镜）
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
