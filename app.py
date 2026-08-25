"""三体宇宙应用：ThreeBodyUniverse（物理推进 + 渲染调度 + GUI + 主循环）。

依赖关系：
    physics  三体系统与初始配置（纯 numpy）
    camera   相机控制（电影运镜 / 自由飞行）
    trails   尾迹环形缓冲与上传
    render   Taichi 渲染管线（render.pipeline 为唯一渲染入口）
"""

import os
import time

import numpy as np
import taichi as ti

from camera import CameraRig
from physics import CONFIG_ORDER, ThreeBodySystem, get_config
from render import pipeline
from render.context import (BLOOM_THR, EXPOSURE_DEF, RES, STAR_TINTS,
                            TAIL_MAX)
from render.state import (cam_fov_f, cam_look_f, cam_pos_f, star_gain_f,
                          star_pos_f, star_rad_f, star_seeds, star_tints)
from trails import TrailBuffer


class ThreeBodyUniverse:
    """三体宇宙：物理推进 + 自绘 HDR 渲染管线 + 交互主循环。"""

    def __init__(self, config_key='pythagorean', res=RES, show_window=True,
                 record_dir=None, record_frames=0):
        self.res = res
        self.show_window = show_window
        self.record_dir = record_dir
        self.record_frames = record_frames

        self.window = ti.ui.Window('Three-Body Universe  ·  三体宇宙',
                                   res, vsync=show_window, show_window=show_window)
        self.canvas = self.window.get_canvas()
        pipeline.ensure_fields(*res)

        # 交互状态
        self.paused = False
        self.speed = 1.0
        self.tail_len = 1500
        self.exposure = EXPOSURE_DEF
        self.bloom_str = 0.85
        self.config_key = config_key
        self.frame_id = 0
        self.fps = 60.0
        self._fps_t = time.perf_counter()
        self.anim_t = 0.0            # 表面动画时钟（小数值，保证 kernel 内 f32 精度）
        self._last_wall = None

        # 尾迹环形缓冲（numpy 侧维护，渲染时展开上传）
        self.trails = TrailBuffer()
        # 相机
        self.cam_rig = CameraRig()

        self.reset(config_key)

    # ------------------------------------------------------------------ setup

    def reset(self, config_key=None):
        """（重新）初始化物理系统与尾迹"""
        if config_key is not None:
            self.config_key = config_key
        cfg = get_config(self.config_key)

        self.sys = ThreeBodySystem(cfg['masses'], cfg['pos'], cfg['vel'])
        self.dt = cfg['dt']
        self.cfg = cfg

        self.trails.reset()
        self.trails.record(self.sys.pos.astype(np.float32))

        # 恒星视觉参数（质量相关的半径/亮度随 set_masses 同步更新）
        self.star_tint_np = np.array(STAR_TINTS, dtype=np.float32)
        star_tints.from_numpy(self.star_tint_np)
        star_pos_f.from_numpy(self.sys.pos.astype(np.float32))
        self.set_masses(cfg['masses'])
        # 红星色调亮度偏低，按亮度归一化增益，保证三颗星视觉亮度接近
        lum = (0.2126 * self.star_tint_np[:, 0]
               + 0.7152 * self.star_tint_np[:, 1]
               + 0.0722 * self.star_tint_np[:, 2])
        gain = np.clip((0.72 / np.maximum(lum, 0.3)) ** 0.75, 0.85, 1.5)
        star_gain_f.from_numpy(gain.astype(np.float32))
        star_seeds.from_numpy(np.array(
            [[0.31, 0.77, 0.19], [0.83, 0.12, 0.55], [0.57, 0.41, 0.92]],
            dtype=np.float32))

        # 相机初始化
        self.cam_rig.snap(self.sys, cfg['cam_dist'])
        self.cam_rig.place_cinematic(0.0, self.sys)
        print(f'[sim3d] 配置: {cfg["name"]}  质量: {np.round(np.asarray(cfg["masses"]), 2).tolist()}')

    def set_masses(self, masses):
        """运行时修改三颗星质量（不重置轨迹）：同步物理与恒星视觉半径。

        质量直接改变引力，轨道会随之演化（这正是“混沌玩法”的乐趣）；
        恒星半径按 0.62*(m/3)^0.38 缩放，辉光/尾迹亮度随半径自动适配。
        """
        m = np.asarray(masses, dtype=np.float64)
        self.sys.masses = m
        self.star_radius = 0.62 * (m / 3.0) ** 0.38
        star_rad_f.from_numpy(self.star_radius.astype(np.float32))

    # ------------------------------------------------------------ simulation

    def advance(self, wall_dt):
        """按时间倍率推进物理，并维护尾迹环形缓冲

        每帧基准模拟时长固定（0.06 模拟秒 @ speed=1）：步数按配置
        dt 归一，近碰撞特解用更细 dt（如 2e-4）时自动加密步数，
        时间流速与手感在不同配置间保持一致。
        """
        if self.paused:
            return
        steps = int(round(60.0 * self.speed * 0.001 / self.dt))
        steps = max(1, min(steps, 1200))
        for _ in range(steps):
            self.sys.step(self.dt)
            self.trails.step(self.dt, self.sys.pos.astype(np.float32))

    # ------------------------------------------------------------------ draw

    def render(self, wall_t):
        """完整渲染管线：上传 -> 场景 -> 尾迹 -> bloom -> tone map -> 呈现。"""
        star_pos_f.from_numpy(self.sys.pos.astype(np.float32))
        cam_pos = np.asarray(self.cam_rig.cam.curr_position, dtype=np.float32)
        cam_look = np.asarray(self.cam_rig.cam.curr_lookat, dtype=np.float32)
        cam_pos_f.from_numpy(np.ascontiguousarray(cam_pos[None, :]))
        cam_look_f.from_numpy(np.ascontiguousarray(cam_look[None, :]))
        cam_fov_f.from_numpy(np.array([self.cam_rig.fov_deg], dtype=np.float32))
        self.trails.upload(self.tail_len)

        pipeline.render_scene(self.anim_t)
        pipeline.splat_trails()
        pipeline.bloom_down(BLOOM_THR)
        pipeline.bloom_blur_h()
        pipeline.bloom_blur_v()
        pipeline.composite(self.exposure, self.bloom_str)
        pipeline.copy_to_texture(pipeline.img_tex)
        self.canvas.set_image(pipeline.img_tex)

    # ------------------------------------------------------------------- gui

    # 左列 GUI 面板布局（归一化坐标，原点左上）；拖拽排除区与之同步
    _GUI_X, _GUI_W = 0.005, 0.26
    _GUI_RECTS = [
        ('Controls',           0.005, 0.355),
        ('Initial conditions', 0.372, 0.295),
        ('Stars & Lens',       0.679, 0.290),
    ]

    def draw_gui(self):
        gui = self.window.get_gui()
        with gui.sub_window('Controls', self._GUI_X, 0.005,
                            self._GUI_W, 0.355) as g:
            g.text(f'Config : {self.cfg["name"]}')
            g.text(f'Time   : {self.sys.time:9.2f}    FPS: {self.fps:5.1f}')
            if 'period' in self.cfg:
                g.text(f'Period : {self.cfg["period"]:8.2f}'
                       f'    Masses: {np.round(self.sys.masses, 1).tolist()}')
            else:
                g.text(f'Masses : {np.round(self.sys.masses, 2).tolist()}')
            g.text('')

            if g.button('Pause / Resume  [SPACE]'):
                self.paused = not self.paused
            if g.button('Reset current  [R]'):
                self.reset()
            g.text('')
            self.speed = g.slider_float('Speed x', self.speed, 0.1, 6.0)
            self.tail_len = g.slider_int('Trail length', self.tail_len,
                                         100, TAIL_MAX)
            self.exposure = g.slider_float('Exposure', self.exposure, 0.3, 2.5)
            self.bloom_str = g.slider_float('Glow (bloom)', self.bloom_str,
                                            0.0, 2.0)
            g.text('')
            label = ('Camera: cinematic  [C]'
                     if self.cam_rig.mode == 'cinematic' else 'Camera: free  [C]')
            if g.button(label):
                self.cam_rig.toggle(self.sys)
            g.text('Free cam: hold RMB + move, WASD / Q E')

        # 初始配置（含三体精确特解，按键 1-8）
        with gui.sub_window('Initial conditions', self._GUI_X, 0.372,
                            self._GUI_W, 0.295) as g:
            for n, key in enumerate(CONFIG_ORDER):
                if g.button(f'[{n + 1}] {get_config(key)["name"]}'):
                    self.reset(key)

        # 质量与镜头控制（可随时调整不重置轨迹）
        with gui.sub_window('Stars & Lens', self._GUI_X, 0.679,
                            self._GUI_W, 0.290) as g:
            g.text('Star masses (live, no reset):')
            newm = self.sys.masses.copy()
            for i in range(3):
                newm[i] = g.slider_float(f'Mass {i + 1}',
                                         float(newm[i]), 0.5, 10.0)
            if not np.allclose(newm, self.sys.masses, rtol=0.0, atol=1e-4):
                self.set_masses(newm)
            if g.button('Reset masses  [M]'):
                self.set_masses(self.cfg['masses'])
            g.text('')
            g.text('Lens / camera:')
            self.cam_rig.fov_deg = g.slider_float(
                'FOV (zoom)  [- =]', self.cam_rig.fov_deg, 15.0, 110.0)
            self.cam_rig.zoom_r = g.slider_float(
                'Orbit dist  [Z X]', self.cam_rig.zoom_r, 0.3, 3.5)
            g.text('Cine: LMB-drag look | Free: RMB+WASD')

        # 拖拽排除区 = 整个左列 GUI 区域（面板布局变化时自动同步）
        self.cam_rig.gui_zone = (self._GUI_X + self._GUI_W + 0.01,
                                 self._GUI_RECTS[-1][1] + self._GUI_RECTS[-1][2])

    # --------------------------------------------------------------- events

    def handle_events(self):
        for e in self.window.get_events(ti.ui.PRESS):
            k = e.key
            if k == ti.ui.SPACE:
                self.paused = not self.paused
            elif k in ('r', 'R'):
                self.reset()
            elif k in ('c', 'C'):
                self.cam_rig.toggle(self.sys)
            elif len(k) == 1 and k in '123456789':
                i = int(k) - 1
                if i < len(CONFIG_ORDER):
                    self.reset(CONFIG_ORDER[i])
            elif k in ('m', 'M'):
                self.set_masses(self.cfg['masses'])
            elif k in ('-', '_'):
                self.cam_rig.zoom_fov(-3.0)     # 长焦：拉近
            elif k in ('=', '+'):
                self.cam_rig.zoom_fov(+3.0)     # 广角：拉远
            elif k in ('z', 'Z'):
                self.cam_rig.dolly(0.88)        # 推近（电影模式）
            elif k in ('x', 'X'):
                self.cam_rig.dolly(1.14)        # 拉远（电影模式）
            elif k == ti.ui.ESCAPE:
                self.window.running = False

    # ------------------------------------------------------------------ main

    def _update_fps(self):
        now = time.perf_counter()
        dt = now - self._fps_t
        if dt > 0.5:
            self.fps = 1.0 / dt
            self._fps_t = now

    def run(self):
        print('[sim3d] 窗口已启动。快捷键: SPACE 暂停 | R 重置 | C 相机 | 1-8 配置'
              '（含三体特解） | M 复位质量 | -/= 变焦 | Z/X 推拉'
              ' | 电影模式拖左键转视角 | ESC 退出')
        try:
            while self.window.running:
                wall_t = time.perf_counter()
                self.handle_events()
                self.advance(wall_t)
                if self._last_wall is not None:
                    self.anim_t += min(wall_t - self._last_wall, 0.1)
                self._last_wall = wall_t
                self.cam_rig.update(wall_t, self.sys, self.speed, self.window)
                self.render(wall_t)
                self.draw_gui()

                if self.record_dir is not None and self.frame_id < self.record_frames:
                    os.makedirs(self.record_dir, exist_ok=True)
                    self.window.save_image(os.path.join(
                        self.record_dir, f'frame_{self.frame_id:04d}.png'))
                self.frame_id += 1
                self._update_fps()

                if self.show_window:
                    self.window.show()
                else:
                    # 离屏模式：达到帧数即退出
                    if self.record_dir is None or self.frame_id >= self.record_frames:
                        break
        finally:
            self.window.destroy()
            print('[sim3d] 窗口已关闭。')


# ============================================================================
# 自测（供 sim3d.py --selftest 调用）
# ============================================================================

def run_selftest():
    """离屏渲染若干时刻的截图，验证渲染管线与视觉效果。"""
    out_dir = 'shots'
    os.makedirs(out_dir, exist_ok=True)
    targets = [3.0, 12.0, 30.0, 55.0]

    app = ThreeBodyUniverse('pythagorean', res=(1280, 800), show_window=False)
    app.speed = 1.0
    print('[selftest] 开始离屏渲染（毕达哥拉斯配置）...')

    shot = 0
    for target in targets:
        # 快速推进物理到目标时刻（期间持续记录尾迹点；
        # 注意每个目标时刻用局部 acc 重新起相位，沿用原实现语义）
        acc = 0.0
        while app.sys.time < target:
            app.sys.step(app.dt)
            acc += app.dt
            if acc >= app.trails.dt:
                acc -= app.trails.dt
                app.trails.record(app.sys.pos.astype(np.float32))
        app.cam_rig.orbit_angle = 0.8 + 0.5 * shot
        app.anim_t = 7.0 * shot
        # 让轨道半径与注视点充分收敛到当前系统尺度（弹射后天体远离质心）
        for _ in range(300):
            app.cam_rig.place_cinematic(shot * 7.0, app.sys)
        app.render(time.perf_counter())
        path = os.path.join(out_dir, f'selftest_t{int(target):02d}.png')
        app.window.save_image(path)
        print(f'[selftest] t={app.sys.time:.1f} -> {path}')
        shot += 1
    app.window.destroy()

    # 输出像素统计便于核对
    from PIL import Image
    for target in targets:
        path = os.path.join(out_dir, f'selftest_t{int(target):02d}.png')
        a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
        lum = a.mean()
        bright = (a.max(axis=2) > 200).mean() * 100
        print(f'[selftest] {path}: mean={lum:.1f} bright_px={bright:.2f}%')
    print('[selftest] 完成。')
