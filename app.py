"""三体宇宙应用：ThreeBodyUniverse（物理推进 + 渲染调度 + GUI + 主循环）。

依赖关系：
    physics  三体系统与初始配置（纯 numpy）
    camera   相机控制（电影运镜 / 自由飞行）
    trails   尾迹环形缓冲与上传
    render   Taichi 渲染管线（render.pipeline 为唯一渲染入口）
"""

import math
import os
import time

import numpy as np
import taichi as ti

from camera import CameraRig
from physics import (CONFIG_ORDER, TYPE_BH, ThreeBodySystem, get_config,
                     star_radius, star_type)
from render import pipeline
from render.context import (BLOOM_THR, EXPOSURE_DEF, RES, STAR_TINTS,
                            TAIL_MAX, TYPE_GAIN, TYPE_SHORT, TYPE_TINTS)
from render.state import (cam_fov_f, cam_look_f, cam_pos_f, star_gain_f,
                          star_mass_f, star_pos_f, star_rad_f, star_seeds,
                          star_tints, star_type_f)
from trails import TrailBuffer


def _remove_imgui_ini():
    """删除 GGUI 的 imgui.ini（面板像素矩形持久化文件）。

    本应用的面板布局每帧按窗口实际尺寸重算，且窗口变形时面板会换新
    窗口 ID 重建（见 draw_gui）；跨会话恢复的过期像素矩形只会导致
    面板错位重叠，属纯污染，直接清理。
    """
    try:
        os.remove('imgui.ini')
    except OSError:
        pass


class ThreeBodyUniverse:
    """三体宇宙：物理推进 + 自绘 HDR 渲染管线 + 交互主循环。"""

    def __init__(self, config_key='pythagorean', res=RES, show_window=True,
                 record_dir=None, record_frames=0):
        self.res = res
        self.show_window = show_window
        self.record_dir = record_dir
        self.record_frames = record_frames

        _remove_imgui_ini()          # 防止跨会话恢复过期的面板像素矩形
        self.window = ti.ui.Window('Three-Body Universe  ·  三体宇宙',
                                   res, vsync=show_window, show_window=show_window)
        self.canvas = self.window.get_canvas()
        pipeline.ensure_fields(*res)

        # GUI 布局状态：窗口形状变化时递增代数（面板窗口 ID 后缀），
        # 因为 GGUI 的 sub_window 矩形仅首次生效（详见 draw_gui）
        self._gui_shape = self.window.get_window_shape()
        self._gui_gen = 0

        # 交互状态
        self.paused = False
        self.speed = 1.0
        self.tail_len = 1500
        self.exposure = EXPOSURE_DEF
        self.bloom_str = 0.85
        self.config_key = config_key
        self.collisions = True     # 恒星碰撞并合（reset 时按配置覆盖）
        self.frame_id = 0
        self.fps = 60.0
        self._fps_t = time.perf_counter()
        self.anim_t = 0.0            # 表面动画时钟（小数值，保证 kernel 内 f32 精度）
        self._last_wall = None
        self._flash = {}             # 星号 -> 并合时刻（临时增亮闪光）
        self._dead_trail = {}        # 死星 -> (末尾点数, 熄灭时刻)，尾迹渐隐用

        # 尾迹环形缓冲（numpy 侧维护，渲染时展开上传）
        self.trails = TrailBuffer()
        # 相机
        self.cam_rig = CameraRig()

        self.reset(config_key)

    # ------------------------------------------------------------------ setup

    def reset(self, config_key=None):
        """（重新）初始化物理系统与尾迹"""
        fresh = False
        if config_key is not None:
            self.config_key = config_key
            fresh = (config_key == 'random')   # 切换到 random 重新采样
        cfg = get_config(self.config_key, fresh=fresh)

        self.sys = ThreeBodySystem(cfg['masses'], cfg['pos'], cfg['vel'])
        self.dt = cfg['dt']
        self.cfg = cfg
        self.collisions = bool(cfg.get('collide', True))
        self._flash = {}
        self._dead_trail = {}

        self.trails.reset()

        # 恒星视觉参数（质量 + 密度 -> 半径/类型/色调/增益，统一在
        # _sync_visual 计算；star_tint_np 为基色，致密天体色由类型覆盖）
        self.star_tint_np = np.array(STAR_TINTS, dtype=np.float32)
        self.density = np.asarray(cfg.get('density', (3.0, 3.0, 3.0)),
                                  dtype=np.float64).copy()
        star_pos_f.from_numpy(self.sys.pos.astype(np.float32))
        self.set_masses(cfg['masses'])
        self.trails.record(self.sys.pos.astype(np.float32),
                           self._trail_alive())   # 需 star_types，在 set_masses 后
        star_seeds.from_numpy(np.array(
            [[0.31, 0.77, 0.19], [0.83, 0.12, 0.55], [0.57, 0.41, 0.92]],
            dtype=np.float32))

        # 相机初始化
        self.cam_rig.snap(self.sys, cfg['cam_dist'])
        self.cam_rig.place_cinematic(0.0, self.sys)
        types = '/'.join(TYPE_SHORT[int(t)] for t in star_type(self.density))
        print(f'[sim3d] 配置: {cfg["name"]}  质量: '
              f'{np.round(np.asarray(cfg["masses"]), 2).tolist()}'
              f'  类型: {types}')

    @staticmethod
    def _gain_for(tints):
        """按色调亮度归一化增益（红星色调亮度低 → 增益补尝）"""
        lum = (0.2126 * tints[:, 0] + 0.7152 * tints[:, 1]
               + 0.0722 * tints[:, 2])
        return np.clip((0.72 / np.maximum(lum, 0.3)) ** 0.75, 0.85, 1.5)

    def _sync_visual(self):
        """把质量 + 密度同步为渲染视觉参数（半径/类型/色调/增益/质量）。

        类型由密度决定（physics.star_type）；半径由 physics.star_radius
        （MS/WD 等密度球、NS 鈐到 ≥2R_s、BH 即 R_s）。有效色调：MS
        用基色（含并合混合色），致密天体用类型色；增益 = 色调亮度
        归一 × 类型增益（NS 极亮喂 bloom）。死星质量/半径为 0，
        渲染与碰撞自动跳过。
        """
        m = self.sys.masses.copy()
        m[~self.sys.alive] = 0.0
        ty = star_type(self.density)
        # 活星转黑洞（密度滑条拉满/致密并合）：旧光迹停止记录并
        # 登记渐隐（黑洞无光，冻结的尾迹会永久滞留成僵直亮线）
        if getattr(self, 'star_types', None) is not None:
            for i in range(3):
                if self.sys.alive[i] and ty[i] == TYPE_BH \
                        and self.star_types[i] != TYPE_BH \
                        and i not in self._dead_trail:
                    self._dead_trail[i] = (int(self.trails.count[i]),
                                           self.sys.time)
        self.star_types = ty
        self.star_radius = star_radius(m, self.density)
        eff = self.star_tint_np.copy()
        for i in range(3):
            t = int(ty[i])
            if t > 0:
                eff[i] = np.array(TYPE_TINTS[t], dtype=np.float32)
        gain = self._gain_for(eff)
        for i in range(3):
            gain[i] *= TYPE_GAIN[int(ty[i])]
        self.star_gain_np = gain
        star_rad_f.from_numpy(self.star_radius.astype(np.float32))
        star_type_f.from_numpy(ty.astype(np.int32))
        star_mass_f.from_numpy(m.astype(np.float32))
        star_tints.from_numpy(eff.astype(np.float32))
        star_gain_f.from_numpy(gain.astype(np.float32))

    def set_masses(self, masses):
        """运行时修改三颗星质量（不重置轨迹）：同步物理质量与视觉参数。

        质量直接改变引力，轨道会随之演化（这正是“混沌玩法”的乐趣）；
        半径/类型/色调/增益由 _sync_visual 按质量 + 密度统一计算。
        已并合熄灭的星质量恒为 0（半径也为 0，渲染自动跳过）。
        """
        m = np.asarray(masses, dtype=np.float64).copy()
        m[~self.sys.alive] = 0.0
        self.sys.masses = m
        self._sync_visual()

    # ------------------------------------------------------------ simulation

    def _trail_alive(self):
        """尾迹记录掩码：活星且非黑洞。

        黑洞自身不发光，不应留光迹（黑洞预设的中央黑洞静止，
        尾迹点会堆在同一像素叠加成亮斑盖住阴影）；其位置由
        吸积盘与阴影充分可视化。
        """
        return self.sys.alive & (self.star_types != TYPE_BH)

    def _step_once(self):
        """单个物理步：积分 + 碰撞检测 + 尾迹采样"""
        step_fn = (self.sys.step_adaptive if self.cfg.get('adaptive')
                   else self.sys.step)
        step_fn(self.dt)
        if self.collisions:
            pair = self.sys.contact_pair(self.star_radius)
            if pair is not None:
                self._apply_merge(*pair)
        self.trails.step(self.dt, self.sys.pos.astype(np.float32),
                         self._trail_alive())

    def _apply_merge(self, i, j):
        """恒星 i 与 j 接触并合：动量守恒地合为一星，j 熄灭。

        视觉同步：色调按质量加权混合、增益按混合色调重算、半径按
        新质量重算；死星尾迹停止记录并随模拟时间从旧端渐隐；
        并合星短暂闪光（碰撞动能耗散为光，近似 nova 式增亮）。
        """
        mi, mj = self.sys.masses[i], self.sys.masses[j]
        m = mi + mj
        self.sys.merge_pair(i, j)

        # 色调按质量加权混合（白热碰撞叠加：向白色靠攏 10%；基色，
        # 致密天体的类型色在 _sync_visual 中覆盖）
        tnew = (mi * self.star_tint_np[i] + mj * self.star_tint_np[j]) / m
        self.star_tint_np[i] = 0.9 * tnew + 0.1 * np.ones(3, np.float32)

        # 致密性守恒：并合不降低密度（黑洞吞并任何天体仍是黑洞；
        # 中子星吞并主序星仍是中子星）
        self.density[i] = max(self.density[i], self.density[j])
        self.density[j] = self.density[i]

        # 半径/类型/色调/增益：质量 + 新密度统一重算
        self.set_masses(self.sys.masses)

        # 死星尾迹渐隐登记（6 模拟秒内从尾端逐点收敛消失）
        self._dead_trail[j] = (int(self.trails.count[j]), self.sys.time)
        # 并合闪光（指数衰减，提升 bloom 呈现碰撞的爆发感）
        self._flash[i] = self.sys.time
        ty = TYPE_SHORT[int(star_type(self.density[i])[()])]
        print(f'[sim3d] t={self.sys.time:7.2f}  恒星{i + 1}+{j + 1}'
              f' 并合  M={m:.2f} -> {ty}  余星{int(self.sys.alive.sum())}')

    def _trail_trim(self):
        """死星尾迹的渐隐修剪量（从最旧端逐点丢弃）"""
        trim = np.zeros(3, dtype=np.int64)
        for k, (cnt0, t0) in self._dead_trail.items():
            trim[k] = int(min(cnt0, cnt0 * (self.sys.time - t0) / 6.0))
        return trim

    def advance(self, wall_dt):
        """按时间倍率推进物理，并维护尾迹环形缓冲

        每帧基准模拟时长固定（0.06 模拟秒 @ speed=1）：步数按配置
        dt 归一。含近碰撞的特解（cfg['adaptive']）用 step_adaptive
        在近距时自动加密子步（远距用满步长，速度与精度兼得）。
        恒星碰撞（可开关）在每个物理步后检测：表面接触即动量
        守恒并合（详见 _apply_merge）。
        """
        if self.paused:
            return
        steps = int(round(60.0 * self.speed * 0.001 / self.dt))
        steps = max(1, min(steps, 1200))
        for _ in range(steps):
            self._step_once()

    # ------------------------------------------------------------------ draw

    def render(self, wall_t):
        """完整渲染管线：上传 -> 场景 -> 尾迹 -> bloom -> tone map -> 呈现。"""
        star_pos_f.from_numpy(self.sys.pos.astype(np.float32))
        cam_pos = np.asarray(self.cam_rig.cam.curr_position, dtype=np.float32)
        cam_look = np.asarray(self.cam_rig.cam.curr_lookat, dtype=np.float32)
        cam_pos_f.from_numpy(np.ascontiguousarray(cam_pos[None, :]))
        cam_look_f.from_numpy(np.ascontiguousarray(cam_look[None, :]))
        cam_fov_f.from_numpy(np.array([self.cam_rig.fov_deg], dtype=np.float32))
        # 增益：基础值 + 并合闪光（指数衰减，喂给 bloom 呈现爆发）
        gain = self.star_gain_np.copy()
        for k, t0 in self._flash.items():
            age = self.sys.time - t0
            if age >= 0.0:
                gain[k] *= 1.0 + 1.7 * math.exp(-age / 0.8)
        star_gain_f.from_numpy(gain.astype(np.float32))
        self.trails.upload(self.tail_len, self._trail_trim())

        pipeline.render_scene(self.anim_t)
        pipeline.splat_trails()
        pipeline.bloom_down(BLOOM_THR)
        pipeline.bloom_blur_h()
        pipeline.bloom_blur_v()
        pipeline.composite(self.exposure, self.bloom_str)
        pipeline.copy_to_texture(pipeline.img_tex)
        self.canvas.set_image(pipeline.img_tex)

    # ------------------------------------------------------------------- gui

    # 面板内容高度标定（px；imgui 实测：标题栏 30 / 文本行距 17 /
    # 按钮与滑条行距 23，另加安全余量；改面板内容后需同步更新）
    _GUI_FULL_H = (376.0, 254.0, 350.0)   # 全内容所需面板高度
    _GUI_COMP_H = (280.0, 254.0, 310.0)   # 紧凑内容（省略空行/提示文本）所需高度
    _GUI_MARGIN = 6.0                     # 屏幕边距 / 面板间隙
    _GUI_COL_MIN_W = 240.0                # 列最小宽度（再窄则文字难以容纳）
    _GUI_SHARE = (0.38, 0.31, 0.31)       # 单列富余高度分配比例

    def _gui_layout(self):
        """按窗口实际尺寸计算三面板矩形（归一化坐标，原点左上）。

        GGUI 的 sub_window 矩形只在窗口首次出现时生效（imgui 的
        FirstUseEver 语义），窗口变形后不会自动跟随 —— 因此 draw_gui
        在形状变化时用 ##代数 后缀换新窗口 ID 重建面板。
        布局策略：高度充足时三面板单列；不足时 Stars & Lens 移到
        右上角成双列；再不足自动省略空行与提示文本；最后等比压缩
        （面板内部滚动/裁剪）——面板矩形在任何窗口尺寸下都不会重叠。
        """
        w, h = self.window.get_window_shape()
        m = self._GUI_MARGIN
        col_w = min(max(0.26 * w, self._GUI_COL_MIN_W), 0.90 * w)
        full, comp = self._GUI_FULL_H, self._GUI_COMP_H
        names = ('Controls', 'Initial conditions', 'Stars & Lens')

        def stack(idxs, heights, x_px):
            rects, y = [], m
            for i, hh in zip(idxs, heights):
                rects.append((names[i], x_px / w, y / h, col_w / w, hh / h))
                y += hh + m
            return rects

        def degrade(comp_h, full_h, avail):
            """avail 不够 full 时的降级：先保 compact 下限，再等比压缩"""
            k = avail / sum(full_h)
            hs = [max(comp_h[i], full_h[i] * k) for i in range(len(full_h))]
            if sum(hs) > avail:
                k2 = avail / sum(comp_h)
                hs = [v * k2 for v in comp_h]
            return hs

        if h >= sum(full) + 4 * m or w < 2 * col_w + 4 * m:
            # ---- 单列（高度充足，或窗口太窄放不下双列） ----
            avail = h - 4 * m
            if avail >= sum(full):
                extra = avail - sum(full)
                hs = [full[i] + extra * s
                      for i, s in enumerate(self._GUI_SHARE)]
            else:
                hs = degrade(comp, full, avail)
            return stack((0, 1, 2), hs, m)

        # ---- 双列：左列 Controls + Initial conditions，右上 Stars & Lens ----
        left_full, left_comp = (full[0], full[1]), (comp[0], comp[1])
        avail = h - 3 * m
        if avail >= sum(left_full):
            hs = list(left_full)
        else:
            hs = degrade(left_comp, left_full, avail)
        rects = stack((0, 1), hs, m)
        rh = min(full[2], h - 2 * m)
        rects.append((names[2], (w - col_w - m) / w, m / h,
                      col_w / w, rh / h))
        return rects

    def draw_gui(self):
        gui = self.window.get_gui()
        # 窗口形状变化 -> 递增代数后缀重建面板（sub_window 矩形仅首次生效）
        shape = self.window.get_window_shape()
        if shape != self._gui_shape:
            self._gui_shape = shape
            self._gui_gen += 1
        tag = f'##{self._gui_gen}' if self._gui_gen else ''

        rects = self._gui_layout()
        h_px = shape[1]
        # 面板高度不足全内容所需 -> 紧凑模式（省略空行与提示文本）
        tight = [r[4] * h_px < f - 2.0
                 for r, f in zip(rects, self._GUI_FULL_H)]

        r = rects[0]
        with gui.sub_window('Controls' + tag, r[1], r[2], r[3], r[4]) as g:
            g.text(f'Config : {self.cfg["name"]}')
            g.text(f'Time   : {self.sys.time:9.2f}    FPS: {self.fps:5.1f}')
            if 'period' in self.cfg:
                g.text(f'Period : {self.cfg["period"]:8.2f}'
                       f'    Masses: {np.round(self.sys.masses, 1).tolist()}')
            else:
                g.text(f'Masses : {np.round(self.sys.masses, 2).tolist()}')
            n_alive = int(self.sys.alive.sum())
            if n_alive < 3:
                g.text(f'Stars  : {n_alive} alive'
                       f'  ({3 - n_alive} merged)')
            self.collisions = g.checkbox('Star collisions (merge)',
                                         self.collisions)
            if not tight[0]:
                g.text('')

            if g.button('Pause / Resume  [SPACE]'):
                self.paused = not self.paused
            if g.button('Reset current  [R]'):
                self.reset()
            if not tight[0]:
                g.text('')
            self.speed = g.slider_float('Speed x', self.speed, 0.1, 6.0)
            self.tail_len = g.slider_int('Trail length', self.tail_len,
                                         100, TAIL_MAX)
            self.exposure = g.slider_float('Exposure', self.exposure, 0.3, 2.5)
            self.bloom_str = g.slider_float('Glow (bloom)', self.bloom_str,
                                            0.0, 2.0)
            if not tight[0]:
                g.text('')
            label = ('Camera: cinematic  [C]'
                     if self.cam_rig.mode == 'cinematic' else 'Camera: free  [C]')
            if g.button(label):
                self.cam_rig.toggle(self.sys)
            if not tight[0]:
                g.text('Free cam: hold RMB + move, WASD / Q E')

        r = rects[1]
        with gui.sub_window('Initial conditions' + tag,
                            r[1], r[2], r[3], r[4]) as g:
            for n, key in enumerate(CONFIG_ORDER):
                # random 显示固定名（避免每帧预生成 200ms 的采样）
                label = ('Random' if key == 'random'
                         else get_config(key)['name'])
                if g.button(f'[{n + 1}] {label}'):
                    self.reset(key)

        r = rects[2]
        with gui.sub_window('Stars & Lens' + tag,
                            r[1], r[2], r[3], r[4]) as g:
            g.text('Star masses (live, no reset):')
            newm = self.sys.masses.copy()
            for i in range(3):
                if self.sys.alive[i]:
                    newm[i] = g.slider_float(f'Mass {i + 1}',
                                             float(newm[i]), 0.5, 10.0)
                else:
                    g.text(f'Mass {i + 1}: merged into'
                           f' {self.sys.host[i] + 1}')
            if not np.allclose(newm, self.sys.masses, rtol=0.0, atol=1e-4):
                self.set_masses(newm)
            if g.button('Reset masses  [M]'):
                if self.sys.alive.all():
                    self.set_masses(self.cfg['masses'])

            g.text('Densities (MS/WD/NS/BH):')
            for i in range(3):
                if self.sys.alive[i]:
                    ty = TYPE_SHORT[int(star_type(self.density[i]))]
                    s = g.slider_float(
                        f'rho {i + 1} [{ty}]',
                        float(np.log10(self.density[i])), -0.3, 6.0)
                    newr = 10.0 ** s
                    if abs(newr - self.density[i]) > 0.02 * self.density[i]:
                        self.density[i] = newr
                        self._sync_visual()
                else:
                    g.text(f'rho {i + 1}: merged into'
                           f' {self.sys.host[i] + 1}')
            if not tight[2]:
                g.text('')
                g.text('Lens / camera:')
            self.cam_rig.fov_deg = g.slider_float(
                'FOV (zoom)  [- =]', self.cam_rig.fov_deg, 15.0, 110.0)
            self.cam_rig.zoom_r = g.slider_float(
                'Orbit dist  [Z X]', self.cam_rig.zoom_r, 0.3, 3.5)
            if not tight[2]:
                g.text('Cine: LMB-drag look | Free: RMB+WASD')

        # 拖拽排除区 = 各面板矩形（每帧随实际布局同步；双列时含右上角）
        pad = 0.012
        self.cam_rig.gui_zones = [
            (r[1] - pad, r[2] - pad, r[1] + r[3] + pad, r[2] + r[4] + pad)
            for r in rects]

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
                if self.sys.alive.all():
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
        print('[sim3d] 窗口已启动。快捷键: SPACE 暂停 | R 重置 | C 相机 | 1-9 配置'
              '（含三体特解与黑洞） | M 复位质量 | -/= 变焦 | Z/X 推拉'
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
            _remove_imgui_ini()
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
        # 快速推进物理到目标时刻（碰撞并合同样生效；_step_once 内
        # 同步维护尾迹采样与碰撞检测）
        while app.sys.time < target:
            app._step_once()
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
    _remove_imgui_ini()

    # 输出像素统计便于核对
    from PIL import Image
    for target in targets:
        path = os.path.join(out_dir, f'selftest_t{int(target):02d}.png')
        a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
        lum = a.mean()
        bright = (a.max(axis=2) > 200).mean() * 100
        print(f'[selftest] {path}: mean={lum:.1f} bright_px={bright:.2f}%')
    print('[selftest] 完成。')
