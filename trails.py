"""尾迹：numpy 侧环形缓冲维护 + 展开上传 GPU（render.state 的 trail field）。

槽位按 MAX_BODIES 预分配（潮汐碎片天生获得自己的尾迹 —— 撕裂的
碎片流会拖着渐隐的光迹沿轨道拉长，是 TDE 最重要的视觉叙事）。
"""

import numpy as np

from render.context import MAX_BODIES, TAIL_MAX


class TrailBuffer:
    """每星独立的轨迹环形缓冲（槽位数 = MAX_BODIES）。

    采样：每累计 sample_dt 模拟时间记录一帧位置（step，只记录
    存活星）；打包（pack）：把每星最近 L_k 个点按时间顺序展开写入
    展开缓冲（pts），并计算每星段数前缀和（prefix，供渲染 kernel
    把 (星, 段) 展平成一维 range 统一调度）。已并合熄灭的星停止
    记录，其历史尾迹由上层通过 trim 参数从旧端逐点修剪，实现平滑
    淡出。GPU 上传由上层统一打包进 staging（见 state.stage）。
    """

    def __init__(self, max_points=TAIL_MAX, sample_dt=0.025):
        self.n_slots = MAX_BODIES
        self.max_points = max_points
        self.dt = sample_dt                   # 每多少模拟时间记录一个轨迹点
        self._buf = np.zeros((MAX_BODIES, max_points, 3), dtype=np.float32)
        self.pts = np.zeros((MAX_BODIES, max_points, 3), dtype=np.float32)
        self.cnt = np.zeros(MAX_BODIES, dtype=np.int32)
        self.prefix = np.zeros(MAX_BODIES + 1, dtype=np.int32)
        self.reset()

    def reset(self):
        self._buf[:] = 0.0
        self.cnt[:] = 0
        self.prefix[:] = 0
        self.head = np.zeros(MAX_BODIES, dtype=np.int64)
        self.count = np.zeros(MAX_BODIES, dtype=np.int64)
        self._acc = 0.0

    def record(self, pos_f32, alive=None):
        """记录当前存活星位置（pos_f32: (n, 3) float32，n ≤ 槽位数）"""
        n = min(len(pos_f32), self.n_slots)
        if alive is None:
            alive = np.ones(n, dtype=bool)
        for k in range(n):
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

    def pack(self, tail_len, trim=None):
        """展开环形缓冲 -> 每星前 L_k 个有效点（填 pts/cnt/prefix）。

        trim: 每星从最旧端额外丢弃的点数（死星尾迹渐隐用）。
        未写入的尾部槽位保留旧值 —— 消费 kernel 以 cnt 为上界，
        不会被读到。
        """
        trim = np.zeros(self.n_slots, dtype=np.int64) if trim is None \
            else np.asarray(trim, dtype=np.int64)
        self.cnt[:] = 0
        self.prefix[:] = 0
        for k in range(self.n_slots):
            L = int(min(self.count[k] - trim[k], tail_len))
            if L >= 2:
                idx = (self.head[k] - L + np.arange(L)) % self.max_points
                self.pts[k, :L] = self._buf[k][idx]
                self.cnt[k] = L
            # 段数 = 点数 - 1（不足 2 点无段）；前缀和供 splat 展平调度
            self.prefix[k + 1] = self.prefix[k] + max(self.cnt[k] - 1, 0)
