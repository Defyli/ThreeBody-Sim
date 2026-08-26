"""物理内核：三体引力 + RK4 积分（float64，保证混沌轨迹精度）+ 初始配置。

纯 numpy 实现，不依赖 Taichi，可独立复用与测试。
"""

import math

import numpy as np

# ============================================================================
# 天体类型体系（密度驱动）
#
# 密度 ρ（模拟单位）决定致密天体类型与半径；质量 m 不变时密度只改
# 半径（碰撞半径）与视觉/透镜效应，引力始终是质点 1/d² —— 与真实
# 引力只依赖质量一致。c² = 30 为风格化选取：m=3 的黑洞 R_s = 0.2、
# 阴影（√27/2·R_s ≈ 2.6 R_s）≈ 0.52，与主序星尺度相当、画面可看。
# ============================================================================
TYPE_MS, TYPE_WD, TYPE_NS, TYPE_BH = 0, 1, 2, 3
TYPE_NAMES = ('main-sequence', 'white dwarf', 'neutron star', 'black hole')
C_LIGHT2 = 30.0            # 光速平方（模拟单位；R_s = 2Gm/c²，G=1）
RHO_WD_MIN = 15.0          # 密度阈值：主序星 -> 白矮星
RHO_NS_MIN = 400.0         # 白矮星 -> 中子星
RHO_BH_MIN = 3.0e4         # 中子星 -> 黑洞（坍缩）


def star_type(rho):
    """密度 -> 天体类型（TYPE_*；向量化）"""
    rho = np.asarray(rho, dtype=np.float64)
    ty = np.where(rho >= RHO_BH_MIN, TYPE_BH,
                  np.where(rho >= RHO_NS_MIN, TYPE_NS,
                           np.where(rho >= RHO_WD_MIN, TYPE_WD, TYPE_MS)))
    return ty.astype(np.int64)


def star_radius(m, rho=3.0):
    """质量 + 密度 -> 半径（视觉与碰撞检测共用；向量化）。

    MS/WD：等密度球 r = (3m/4πρ)^⅓（ρ=3 时与旧视觉公式几乎一致）；
    NS：表面不得落入自身视界内，钳到 ≥ 2·R_s（真实中子星 R/R_s ≈ 2.7，
    且所有中子星半径几乎相同 —— 钳位本身就是物理事实的近似）；
    BH：半径即史瓦西半径 R_s = 2m/c²，渲染据此做视界捕获/盘几何。
    m=0（已并合熄灭）-> r=0，自动退出碰撞与渲染。
    """
    m = np.asarray(m, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    ty = star_type(rho)
    rs = 2.0 * m / C_LIGHT2
    r = np.maximum((3.0 * m / (4.0 * np.pi * np.maximum(rho, 1e-3)))
                   ** (1.0 / 3.0), 0.02)
    r = np.where(ty == TYPE_BH, rs, r)
    r = np.where(ty == TYPE_NS, np.maximum(r, 2.0 * rs), r)
    return r


class ThreeBodySystem:
    """三体引力系统，RK4 积分。状态用 numpy float64 维护。

    支持恒星碰撞并合（merge_pair）：两星接触时动量守恒地合为一星，
    被吞星质量归零并标记死亡（alive=False）；死亡星不再参与引力与
    碰撞检测，位置每步同步到宿主星（作为“幽灵”随行，_accel 中
    行/列均已掩蔽，不会产生数值影响）。
    """

    def __init__(self, masses, positions, velocities, G=1.0):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.pos = np.asarray(positions, dtype=np.float64).copy()
        self.vel = np.asarray(velocities, dtype=np.float64).copy()
        self.G = float(G)
        self.time = 0.0
        self.alive = np.ones(3, dtype=bool)       # 恒星存活状态
        self.host = np.full(3, -1, dtype=np.int64)  # 死星 -> 并合宿主

    def _accel(self, pos):
        """向量化引力加速度: a_i = sum_j G m_j (r_j - r_i) / |r_j - r_i|^3"""
        d = pos[None, :, :] - pos[:, None, :]          # d[i, j] = r_j - r_i
        dist2 = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dist2, 1.0)                    # 防止 0^-1.5
        with np.errstate(divide='ignore'):              # 幽灵同位: 0^-1.5=inf
            inv_d3 = dist2 ** -1.5
        np.fill_diagonal(inv_d3, 0.0)
        inv_d3[dist2 == 0.0] = 0.0                      # 幽灵与宿主同位对
        # 死星不施力（质量已为 0，此处同时避免幽灵与宿主同位时
        # 0·inf = NaN）；也不受力（位置随后被宿主覆盖）
        dead = ~self.alive
        if dead.any():
            inv_d3[:, dead] = 0.0
            inv_d3[dead, :] = 0.0
        return self.G * np.einsum('ij,ijk->ik',
                                  self.masses[None, :] * inv_d3, d)

    def step(self, dt):
        """一个经典 RK4 步长"""
        p, v = self.pos, self.vel
        k1v = self._accel(p)
        k2v = self._accel(p + 0.5 * dt * v)
        k3v = self._accel(p + 0.5 * dt * (v + 0.5 * dt * k1v))
        k4v = self._accel(p + dt * (v + 0.5 * dt * k2v))
        self.pos = p + dt * (v + dt / 6.0 * (k1v + k2v + k3v))
        self.vel = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        self._sync_ghosts()
        self.time += dt

    def _sync_ghosts(self):
        """死星位置/速度同步到宿主星（幽灵随行）"""
        for k in np.nonzero(self.host >= 0)[0]:
            h = self.host[k]
            self.pos[k] = self.pos[h]
            self.vel[k] = self.vel[h]

    def merge_pair(self, i, j):
        """恒星碰撞并合：i 存活（动量守恒），j 熄灭。

        位置取质心、速度取动量守恒、质量相加——碰撞动能耗散
        （物理并合的合理近似）。返回 (i, j) 供上层同步视觉状态。
        """
        mi, mj = self.masses[i], self.masses[j]
        m = mi + mj
        self.pos[i] = (mi * self.pos[i] + mj * self.pos[j]) / m
        self.vel[i] = (mi * self.vel[i] + mj * self.vel[j]) / m
        self.masses[i] = m
        self.masses[j] = 0.0
        self.alive[j] = False
        self.host[j] = i
        self.host[self.host == j] = i      # 旧宿主链改指新宿主

    def contact_pair(self, radii):
        """首对表面接触的活星（d < r_i + r_j），无则返回 None"""
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if self.alive[i] and self.alive[j]:
                d = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if d < radii[i] + radii[j]:
                    return i, j
        return None

    def min_pair_dist(self):
        """最近两活星间距（自适应步长的加密依据；死星幽灵不参与）"""
        p, a = self.pos, self.alive
        cand = [float(np.linalg.norm(p[i] - p[j]))
                for i, j in ((0, 1), (0, 2), (1, 2)) if a[i] and a[j]]
        return min(cand) if cand else 1e9

    def step_adaptive(self, dt, kappa=4.7e-3, max_sub=48):
        """自适应子步 RK4：近距按引力时标 κ·d^1.5 加密。

        固定细步长对含近碰撞的周期解（Šuvakov 族）在远距阶段浪费
        大量步数；本方法只在近距时加密（近距动力学时标 ~ d^1.5），
        远距用满步长——含近碰撞轨道下速度与精度同时优于固定步长。
        κ 标定：d=0.1 时子步 ≈1.5e-4（实测能量漂移 <1e-5/周期）。
        """
        rem = float(dt)
        n = 0
        while rem > 1e-12:
            if n >= max_sub:
                self.step(rem)            # 兜底：极端穿透时一次收尾
                return
            d = max(self.min_pair_dist(), 0.02)
            h = min(rem, kappa * d ** 1.5)
            self.step(h)
            rem -= h
            n += 1

    @property
    def center_of_mass(self):
        w = self.masses / self.masses.sum()
        return self.pos.T @ w

    @property
    def spread(self):
        """系统相对质心的最大半径（用于自适应相机距离）"""
        r = np.linalg.norm(self.pos - self.center_of_mass, axis=1)
        return float(r.max())


# ============================================================================
# 初始配置
#
# 除毕达哥拉斯（混沌）与随机外，其余均为三体问题的精确周期特解：
#   figure8      Chenciner–Montgomery 八字形解（2000，线性稳定）
#   equilateral  拉格朗日等边三角形解（1772）：三体绕质心刚性旋转，
#                ω = sqrt(G·M/a³)；等质量下线性不稳定，运行数十个
#                周期后可见对称性缓慢破缺——本身就是可看的物理
#   euler        欧拉共线解（1767）：三点一线绕质心自转，
#                等质量时 ω² = 5m/(4a³)，中体恰在质心不动（不稳定）
#   butterfly / moth / yin_yang
#                Šuvakov–Dmitrašinović 等质量周期解族（PRL 110, 114301,
#                2013）；原始尺度 ±1 太小，整体放大 _SD_SCALE 倍；含极
#                近碰撞，配 adaptive=True 自适应子步（见 step_adaptive）
# ============================================================================

_RANDOM_TRIES = 96          # 随机配置拒绝采样上限
_RANDOM_SPAN = 2.0          # 位置散布（±span 均匀采样；散布越大动力
                            # 学时标越长，密接前互动越充分）
_RANDOM_GAP = 2.2           # 出生最小间距（×两星半径和）
_RANDOM_SURVIVE = 6.0       # 出生后免并合预演时长（模拟时间单位；
                            # 实测三体密接时标 ~1-4，过长则无解）


def _random_try(rng):
    """单次随机采样 + 两道校验，不合格返回 None（供拒绝采样循环）。

    速度按维里尺度缩放：v_rms ~ sqrt(G·M/(2·d_mean))×U(0.55,1.05)。
    纯均匀小速度（旧版 ±0.45）远低于维里值，系统深束缚、快速
    自由落体坍缩，几平必然立刻密接并合；维里附近的速度让系统
    处于轨道互动状态，密接显著延后（实测首撞中位 12 单位）。
    密度取 10^U(0.3,0.7) ∈ [2,5]，均为普通主序星（随机玩法不
    混入致密天体，想要黑洞清切 blackhole 预设或调密度滑条）。
    """
    masses = rng.uniform(0.8, 1.6, 3)
    rho = 10.0 ** rng.uniform(0.3, 0.7)
    pos = rng.uniform(-_RANDOM_SPAN, _RANDOM_SPAN, (3, 3))
    vel = rng.uniform(-1.0, 1.0, (3, 3))          # 方向载体（缩放前）
    d_mean = np.mean([np.linalg.norm(pos[i] - pos[j])
                      for i, j in ((0, 1), (0, 2), (1, 2))])
    v_scale = np.sqrt(masses.sum() / (2.0 * d_mean))
    vrms = np.sqrt((vel ** 2).sum() / 3.0)
    vel *= (v_scale * rng.uniform(0.55, 1.05)) / max(vrms, 1e-9)
    vel -= (vel * masses[:, None]).sum(0) / masses.sum()   # 去质心漂移
    radii = star_radius(masses, rho)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        gap = np.linalg.norm(pos[i] - pos[j]) / (radii[i] + radii[j])
        if gap < _RANDOM_GAP:
            return None
    probe = ThreeBodySystem(masses, pos, vel)   # 间距合格 -> 预演存活
    while probe.time < _RANDOM_SURVIVE:
        probe.step_adaptive(0.004)
        if probe.contact_pair(radii) is not None:
            return None
    return dict(name='Random', masses=masses, pos=pos, vel=vel,
                density=[rho] * 3, dt=0.001, cam_dist=8.0)


def _random_config():
    """随机配置（拒绝采样）：保证出生不贴脸、短期内不并合。

    位置纯均匀采样时两星出生间距小于半径和（直接并合）或貒脸
    （接触即并合）的概率不低 —— 碰撞并合开启的当下会“一出生
    就并合”。生成后做两道校验（见 _random_try），不满足则重采样：
      1) 所有两两间距 ≥ _RANDOM_GAP×半径和（出生无接触且留裕度）；
      2) 自适应步长预演 _RANDOM_SURVIVE 个模拟时间单位无并合
         —— 观众至少能先看到一段三体互动。
    全部失败时兑底：最后一次采样绕质心径向放大到刚好满足间距
    裕度（引力时标随之变长，几乎必然存活）。default_rng() 无参
    播种用系统熵，快速连续调用不会重复种子。
    """
    rng = np.random.default_rng()
    for _ in range(_RANDOM_TRIES):
        cfg = _random_try(rng)
        if cfg is not None:
            return cfg
    # 兑底：新采样绕质心径向放大至满足间距裕度（相对质心缩放对
    # 两两间距严格成立：p'_i - p'_j = scale·(p_i - p_j)；间距拉开
    # 后引力时标变长，短期内几乎必然不再并合）
    masses = rng.uniform(0.8, 1.6, 3)
    pos = rng.uniform(-_RANDOM_SPAN, _RANDOM_SPAN, (3, 3))
    vel = rng.uniform(-0.45, 0.45, (3, 3))
    vel -= (vel * masses[:, None]).sum(0) / masses.sum()
    rho = 3.0
    com = (masses[:, None] * pos).sum(0) / masses.sum()
    radii = star_radius(masses, rho)
    scale = 1.0
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = max(np.linalg.norm(pos[i] - pos[j]), 1e-9)
        scale = max(scale, 1.15 * _RANDOM_GAP * (radii[i] + radii[j]) / d)
    pos = com + (pos - com) * scale
    return dict(name='Random', masses=masses, pos=pos, vel=vel,
                density=[rho] * 3, dt=0.001, cam_dist=8.0)


_SD_SCALE = 3.0     # Šuvakov 解整体尺度（长度单位）


def _suvakov(name, vx, vy, period, cam_dist=7.0):
    """Šuvakov–Dmitrašinović 族初值构造。

    原始解（G = m = 1）：x1=(-1,0), x2=(1,0), x3=(0,0)，
    v1 = v2 = (vx,vy)，v3 = -2(vx,vy)。引力体系无内禀尺度：
    长度放大 s 时速度按 s^-1/2、周期按 s^3/2 缩放后仍为精确解。

    注意：这族解都含极近碰撞（原始尺度最小间距 ~0.01-0.08，
    是其周期性的物理本质），因此配合 adaptive=True：常规阶段
    用 dt=1e-3，近距时由 step_adaptive 按引力时标自动加密
    （能量漂移实测 <1e-5/周期，比固定 2e-4 更准且快约 5 倍）。
    """
    c = _SD_SCALE ** -0.5
    return dict(
        name=name, masses=[1.0, 1.0, 1.0],
        pos=[[-_SD_SCALE, 0.0, 0.0], [_SD_SCALE, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        vel=[[vx * c, vy * c, 0.0], [vx * c, vy * c, 0.0],
             [-2.0 * vx * c, -2.0 * vy * c, 0.0]],
        dt=0.001, cam_dist=cam_dist, adaptive=True,
        period=period * _SD_SCALE ** 1.5, collide=False)


CONFIGS = {
    'pythagorean': dict(
        name='Pythagorean (3-3-3)',
        masses=[3.0, 3.0, 3.0],
        pos=[[1.0, 3.0, 0.0], [-2.0, -1.0, 0.0], [1.0, -1.0, 0.0]],
        vel=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dt=0.001, cam_dist=16.0,
    ),
    'figure8': dict(
        name='Figure-8 orbit',
        masses=[1.0, 1.0, 1.0],
        pos=[[-0.97000436, 0.24308753, 0.0],
             [0.97000436, -0.24308753, 0.0],
             [0.0, 0.0, 0.0]],
        vel=[[0.4662036850, 0.4323657300, 0.0],
             [0.4662036850, 0.4323657300, 0.0],
             [-0.9324073700, -0.8647314600, 0.0]],
        dt=0.001, cam_dist=4.2, period=6.325913, collide=False,
    ),
    'equilateral': dict(
        name='Lagrange triangle',
        masses=[1.0, 1.0, 1.0],
        # 边长 a=3 的等边三角形，绕质心刚性旋转：ω = sqrt(3/a³) = 1/3；
        # v_i = ω·ẑ×r_i（逆时针），质心与总动量均为零
        pos=[[0.0, math.sqrt(3.0), 0.0],
             [-1.5, -math.sqrt(3.0) / 2.0, 0.0],
             [1.5, -math.sqrt(3.0) / 2.0, 0.0]],
        vel=[[-1.0 / math.sqrt(3.0), 0.0, 0.0],
             [0.5 / math.sqrt(3.0), -0.5, 0.0],
             [0.5 / math.sqrt(3.0), 0.5, 0.0]],
        dt=0.001, cam_dist=5.5, period=6.0 * math.pi, collide=False,
    ),
    'euler': dict(
        name='Euler collinear',
        masses=[1.0, 1.0, 1.0],
        # 等质量共线：外侧两体位于 ±a，ω² = 5m/(4a³)（a=2 -> ω=sqrt(5/32)），
        # 中体恰在质心、所受合力为零；外侧速度大小 = ωa = sqrt(5/8)
        pos=[[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vel=[[0.0, -math.sqrt(5.0 / 8.0), 0.0],
             [0.0, 0.0, 0.0],
             [0.0, math.sqrt(5.0 / 8.0), 0.0]],
        dt=0.001, cam_dist=6.0,
        period=2.0 * math.pi * math.sqrt(32.0 / 5.0), collide=False,
    ),
    'butterfly': _suvakov('Butterfly I', 0.30689, 0.12551, 6.2356),
    'moth': _suvakov('Moth I', 0.46444, 0.39606, 14.8939),
    'yin_yang': _suvakov('Yin-Yang I', 0.51394, 0.30474, 17.3284),
    'blackhole': dict(
        # 黑洞 + 双星：中央 m=6 黑洞（R_s=0.4，阴影 ~1.0，吸积盘
        # 1.2~3.6），两颗 m=1.2 主序星对称绕行 —— 共线中心构型的
        # 精确相对平衡解（欧拉型）：v² = G(M_bh+m/4)/r。
        # 双星在盘外缘（r=5 > 3.6）稳定绕行，失稳被吞时触发并合。
        name='Black hole + 2 stars',
        masses=[6.0, 1.2, 1.2],
        pos=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]],
        vel=[[0.0, 0.0, 0.0], [0.0, 1.122, 0.0], [0.0, -1.122, 0.0]],
        density=[3.0e5, 3.0, 3.0],
        dt=0.001, cam_dist=16.0,
    ),
    'random': None,   # 动态生成
}

CONFIG_ORDER = ['pythagorean', 'figure8', 'equilateral', 'euler',
                'butterfly', 'moth', 'yin_yang', 'blackhole', 'random']


def get_config(key, fresh=False):
    """取初始配置；random 在 fresh=True 时重新采样。

    random 的两种语义：切换到 random（按钮/数字键）重新随机；
    R 键重置当前则复现同一次采样（fresh=False，可重现当前轨道）。
    """
    cfg = CONFIGS[key]
    if cfg is None or (fresh and key == 'random'):
        cfg = _random_config()
        CONFIGS[key] = cfg
    return cfg
