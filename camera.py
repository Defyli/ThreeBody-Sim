"""相机控制：电影环绕运镜 <-> GGUI 自由飞行（WASD + 鼠标右键）。

封装 ti.ui.Camera 的状态与两种模式的切换逻辑；运镜参数沿用原实现。
电影模式下支持运行时镜头控制：
    fov_deg  视场角（度）—— 缩放/变焦（GUI 滑条或 - / = 键）
    zoom_r   环绕半径缩放倍率 —— 推拉镜头（GUI 滑条或 Z / X 键）
    按住左键拖拽 —— 手动转动视角（方位 + 俯仰偏移，叠加在自动运镜上）
"""

import math
import time

import numpy as np
import taichi as ti

from render.context import FOV_DEG


class CameraRig:
    """相机装置：mode 为 'cinematic'（自动环绕）或 'free'（用户控制）。"""

    def __init__(self, cam_dist=16.0):
        self.mode = 'cinematic'
        self.orbit_angle = 0.8
        self._orbit_r = float(cam_dist)
        self.lookat = np.zeros(3)
        self.cam = ti.ui.Camera()
        self.cam.up(0.0, 1.0, 0.0)

        # 运行时镜头控制（两种模式共用 fov_deg）
        self.fov_deg = float(FOV_DEG)   # 视场角（度）：15 长焦 ~ 110 广角
        self.zoom_r = 1.0               # 电影模式环绕半径缩放（推拉）
        self._elev_off = 0.0            # 拖拽附加的俯仰偏移（弧度）
        self._drag_pos = None           # 拖拽中上一帧光标（归一化坐标）
        # GUI 面板占用的矩形列表 [(x0, y0, x1, y1)]（归一化，左上原点）：
        # 这些区域内不触发拖拽转向，避免与控件操作冲突；
        # 由 app.draw_gui 每帧按实际布局同步（窗口变形时可能多列）。
        self.gui_zones = [(0.0, 0.0, 0.29, 0.80)]

    # ------------------------------------------------------------------ api

    def snap(self, sys, orbit_r):
        """重置注视点与环绕半径，并复位手动镜头偏移（切换初始配置后调用）。"""
        self.lookat = sys.center_of_mass.copy()
        self._orbit_r = float(orbit_r)
        self._elev_off = 0.0
        self.zoom_r = 1.0

    def update(self, wall_t, sys, speed, window):
        """每帧更新相机（电影运镜或接收用户输入）。"""
        if self.mode == 'cinematic':
            self.orbit_angle += 0.0035 * speed ** 0.3 + 0.0022
            self._steer(window)
            self.place_cinematic(wall_t, sys)
        else:
            self.cam.track_user_inputs(window, movement_speed=8.0,
                                       hold_key=ti.ui.RMB)

    def dolly(self, factor):
        """推拉镜头：缩放环绕半径（<1 拉近，>1 拉远）。"""
        self.zoom_r = float(np.clip(self.zoom_r * factor, 0.3, 3.5))

    def zoom_fov(self, delta_deg):
        """变焦：调整视场角（负值为拉近/长焦，正值为拉远/广角）。"""
        self.fov_deg = float(np.clip(self.fov_deg + delta_deg, 15.0, 110.0))

    def toggle(self, sys, wall_t=None):
        """切换相机模式；切回电影模式时立刻按当前时间放置相机。"""
        if self.mode == 'cinematic':
            self.mode = 'free'
        else:
            self.mode = 'cinematic'
            self.place_cinematic(wall_t if wall_t is not None
                                 else time.perf_counter(), sys)

    # -------------------------------------------------------------- cinematic

    def _steer(self, window):
        """电影模式下按住左键拖拽转动视角（方位 + 俯仰偏移）。

光标坐标为窗口归一化坐标（原点左上、y 向下），与 GUI 子窗口同一
约定；拖拽方向与主流轨道控件一致（拖右场景右转，拖下相机升高）。
self.gui_zones 所围区域（GUI 面板）不触发拖拽，避免与控件冲突；
面板布局变化时由 app.draw_gui 每帧同步该列表。
离屏窗口不支持光标查询，直接跳过。
"""
        try:
            x, y = window.get_cursor_pos()
        except RuntimeError:
            self._drag_pos = None
            return
        in_gui = any(z[0] <= x <= z[2] and z[1] <= y <= z[3]
                     for z in self.gui_zones)
        if window.is_pressed(ti.ui.LMB) and not in_gui:
            if self._drag_pos is not None:
                dx = x - self._drag_pos[0]
                dy = y - self._drag_pos[1]
                self.orbit_angle -= dx * 3.2
                self._elev_off = min(1.25, max(-1.25, self._elev_off + dy * 2.2))
            self._drag_pos = (x, y)
        else:
            self._drag_pos = None

    def place_cinematic(self, wall_t, sys):
        """电影运镜：绕质心缓慢环绕，半径随系统尺度平滑自适应 + 呼吸起伏"""
        spread = sys.spread
        target_r = float(np.clip(spread * 2.9 + 3.5, 6.0, 90.0))
        self._orbit_r += (target_r - self._orbit_r) * 0.02
        breathe = 1.0 + 0.07 * math.sin(wall_t * 0.21)
        r = self._orbit_r * breathe * self.zoom_r
        elev = 0.34 + 0.16 * math.sin(wall_t * 0.09) + self._elev_off
        elev = max(-1.25, min(1.35, elev))
        a = self.orbit_angle
        com = sys.center_of_mass
        self.lookat += (com - self.lookat) * 0.06
        pos = self.lookat + r * np.array([
            math.cos(elev) * math.cos(a), math.sin(elev),
            math.cos(elev) * math.sin(a)])
        self.cam.position(*pos.astype(float))
        self.cam.lookat(*self.lookat.astype(float))
        self.cam.up(0.0, 1.0, 0.0)
