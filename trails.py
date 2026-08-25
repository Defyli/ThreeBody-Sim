"""尾迹：numpy 侧环形缓冲维护 + 展开上传 GPU（render.state 的 trail field）。"""

import numpy as np

from render.context import TAIL_MAX
from render.state import trail_cnt, trail_pts


class TrailBuffer:
    """三颗星的轨迹环形缓冲。

    采样：每累计 sample_dt 模拟时间记录一帧位置（step）；
    上传：把最近 tail_len 个点按时间顺序展开写入 trail field（upload）。
    """

    def __init__(self, max_points=TAIL_MAX, sample_dt=0.025):
        self.max_points = max_points
        self.dt = sample_dt                   # 每多少模拟时间记录一个轨迹点
        self._buf = np.zeros((3, max_points, 3), dtype=np.float32)
        self._upload = np.zeros((3, max_points, 3), dtype=np.float32)
        self.reset()

    def reset(self):
        self._buf[:] = 0.0
        self.head = 0
        self.count = 0
        self._acc = 0.0

    def record(self, pos_f32):
        """记录三颗星当前位置（pos_f32: (3, 3) float32）"""
        self._buf[:, self.head, :] = pos_f32
        self.head = (self.head + 1) % self.max_points
        self.count = min(self.count + 1, self.max_points)

    def step(self, dt, pos_f32):
        """推进 dt 模拟时间，按采样间隔记录轨迹点"""
        self._acc += dt
        if self._acc >= self.dt:
            self._acc -= self.dt
            self.record(pos_f32)

    def upload(self, tail_len):
        """展开环形缓冲 -> 前 L 个有效点，上传 GPU。"""
        L = min(self.count, tail_len)
        if L < 2:
            trail_cnt.from_numpy(np.zeros(3, dtype=np.int32))
            return
        idx = (self.head - L + np.arange(L)) % self.max_points
        for i in range(3):
            self._upload[i, :L] = self._buf[i][idx]
        trail_pts.from_numpy(self._upload)
        trail_cnt.from_numpy(np.array([L, L, L], dtype=np.int32))
