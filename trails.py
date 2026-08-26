"""尾迹：numpy 侧环形缓冲维护 + 展开上传 GPU（render.state 的 trail field）。"""

import numpy as np

from render.context import TAIL_MAX
from render.state import trail_cnt, trail_pts


class TrailBuffer:
    """三颗星的轨迹环形缓冲（每星独立计数）。

    采样：每累计 sample_dt 模拟时间记录一帧位置（step，只记录
    存活星）；上传：把每星最近 L_k 个点按时间顺序展开写入 trail
    field（upload）。已并合熄灭的星停止记录，其历史尾迹由上层
    通过 trim 参数从旧端逐点修剪，实现平滑淡出。
    """

    def __init__(self, max_points=TAIL_MAX, sample_dt=0.025):
        self.max_points = max_points
        self.dt = sample_dt                   # 每多少模拟时间记录一个轨迹点
        self._buf = np.zeros((3, max_points, 3), dtype=np.float32)
        self._upload = np.zeros((3, max_points, 3), dtype=np.float32)
        self.reset()

    def reset(self):
        self._buf[:] = 0.0
        self.head = np.zeros(3, dtype=np.int64)
        self.count = np.zeros(3, dtype=np.int64)
        self._acc = 0.0

    def record(self, pos_f32, alive=None):
        """记录当前存活星位置（pos_f32: (3, 3) float32）"""
        if alive is None:
            alive = np.ones(3, dtype=bool)
        for k in range(3):
            if alive[k]:
                self._buf[k, self.head[k], :] = pos_f32[k]
                self.head[k] = (self.head[k] + 1) % self.max_points
                self.count[k] = min(self.count[k] + 1, self.max_points)

    def step(self, dt, pos_f32, alive=None):
        """推进 dt 模拟时间，按采样间隔记录轨迹点"""
        self._acc += dt
        if self._acc >= self.dt:
            self._acc -= self.dt
            self.record(pos_f32, alive)

    def upload(self, tail_len, trim=None):
        """展开环形缓冲 -> 每星前 L_k 个有效点，上传 GPU。

        trim: 每星从最旧端额外丢弃的点数（死星尾迹渐隐用）。
        """
        trim = np.zeros(3, dtype=np.int64) if trim is None else trim
        cnt = np.zeros(3, dtype=np.int32)
        for k in range(3):
            L = int(min(self.count[k] - trim[k], tail_len))
            if L >= 2:
                idx = (self.head[k] - L + np.arange(L)) % self.max_points
                self._upload[k, :L] = self._buf[k][idx]
                cnt[k] = L
        trail_pts.from_numpy(self._upload)
        trail_cnt.from_numpy(cnt)
