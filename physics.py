"""物理内核：N 体引力（初始三体，潮汐瓦解可增生碎片）+ RK4 积分
（float64，保证混沌轨迹精度）+ 初始配置。

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

MAX_BODIES = 28            # 天体槽位上限（初始 3 + 潮汐碎片；渲染/物理预算）

# ---- 后牛顿动力学常量（1PN + 2.5PN，始终启用） ----
_C_INV_LIGHT2 = 1.0 / C_LIGHT2            # c⁻²（1PN）
_C_INV_LIGHT5 = C_LIGHT2 ** -2.5          # c⁻⁵（2.5PN 引力波辐射）
_PN_GW_K = 8.0 / 15.0                     # 四极光度系数 8/15
_PN_MIN_MASS = 0.5                         # PN 星对质量门槛（碎片贡献 ~m 可忽略）

# ---- 潮汐物理（变形 + 瓦解；见 tidal_state / encounter_scan / disrupt） ----
_TIDE_Q = 2.2              # 洛希系数：d_R = q·r_i·(m_j/m_i)^{1/3}
                            # （流体卫星 2.44、刚体 1.26；恒星为流体，取近流体值）
_TIDE_FLOOR = 0.25         # 可被瓦解的最小质量（更小者为不可再撕的“颗粒”，防级联）
_TIDE_GRACE = 1.2          # 碎片出生保护期：免互撞/免再撕，让潮汐流先剪切散开
_TIDE_MAXFRAG = 5          # 单次瓦解碎片数上限（渐进式：少而大，级联逐层细化）
_TIDE_STRIP_MAX = 1.30     # 剥离带上限（洛希深度 1 < d_R/d ≤ 此值 → 逐层剥离而非粉碎）
_TIDE_PEEL = 0.20          # 每次剥离的质量占比（外围包层 -> 2 颗不可再撕的碎屑）


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
    """N 体引力系统（初始三体；潮汐瓦解可增生碎片至 MAX_BODIES），
    RK4 积分。状态用 numpy float64 维护。

    动力学（始终启用，无开关）：牛顿引力 + 1PN + 2.5PN，
    详见 _accel 的公式与适用范围注释。

    结构性事件（每步由上层经 encounter_scan 驱动）：
      碰撞并合（merge_pair）：两星接触时动量守恒地合为一星，被吞星
      质量归零并标记死亡（alive=False）；死亡星不再参与引力与碰撞
      检测，位置每步同步到宿主星（“幽灵”随行，_accel 中行/列均已
      掩蔽，不会产生数值影响）。
      潮汐瓦解（disrupt）：天体越过另一天体的洛希极限时撕裂为
      碎片（真实粒子，参与引力/碰撞/尾迹）；碎片出生后有保护期
      （_TIDE_GRACE）免互撞/免再撕 —— 刚撕裂的碎片流仍在相互穿越，
      靠开普勒剪切与膨胀速度自然散开后再恢复碰撞（回落吸积成团）。
    """

    def __init__(self, masses, positions, velocities, G=1.0):
        self.masses = np.asarray(masses, dtype=np.float64)
        n = len(self.masses)
        self.pos = np.asarray(positions, dtype=np.float64).copy()
        self.vel = np.asarray(velocities, dtype=np.float64).copy()
        self.G = float(G)
        self.time = 0.0
        self.alive = np.ones(n, dtype=bool)       # 恒星存活状态
        self.host = np.full(n, -1, dtype=np.int64)  # 死星 -> 并合宿主
        self.birth = np.full(n, -1e9)              # 出生时刻（碎片保护期判据）
        self.cohort = np.zeros(n, dtype=np.int64)  # 出生批次（兄弟碎片互撞保护）
        self._eid = 0                              # 瓦解事件计数（cohort id）
        self._pn_idx = ()                          # PN 星对索引（refresh 维护）
        self.refresh()

    def _accel(self, pos, vel):
        """总加速度：牛顿 + 1PN（谐和规范）+ 2.5PN（引力波辐射反作用）。

        牛顿项：a_i = sum_j G m_j (r_j - r_i) / |r_j - r_i|^3（向量化）。

        1PN（逐对叠加的两体精确项，谐和规范相对加速度）：
          a_1PN = GM/(c²r²)·[n·κ + (4-2ν)·ṙ·v]，
          κ = 2(2+ν)GM/r - (1+3ν)v² + (3/2)ν·ṙ²，
          其中 r/v 为相对位矢/速度，ν = m_i·m_j/M²。
          两体时精确（含近日点进动 6πGM/(c²a(1-e²)) 与圆轨频率
          Ω² = GM/r³·[1-(3-ν)GM/(c²r)]）；三体的 1PN 交叉项（同阶
          但通常更小）已省略。按 m_j/M 与 m_i/M 反号分摊，总动量
          严格守恒。

        2.5PN（引力波辐射反作用，能量平衡拖曳）：对每个星对用瞬时
         四极光度 P = (8/15)·G³m_i²m_j²/(c⁵r⁴)·(12v²-11ṙ²) 沿相对
         速度拖曳 -P/(μ_red·v²)·v —— 圆轨时严格等价 Peters 衰减
          ȧ = -(64/5)G³m_i m_j M/(c⁵a³)，偏心轨的轨道平均 ȧ 与
          Peters-Mathews 一致（ė 为近似）；多对引力波相干叠加以
          逐对近似（星对分离时良好）。

        注：本宇宙 c²=30 为风格化小值，普通三星系统的 PN 修正
        ~百分之几 —— 周期特解（八字形等牛顿精确解）会缓慢退相
        位并微幅旋近，这是该宇宙自洽的相对论后果而非 bug。
        """
        # ---- 牛顿项（向量化，与历史版本一致） ----
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
        acc = self.G * np.einsum('ij,ijk->ik',
                                 self.masses[None, :] * inv_d3, d)
        self._pn_pairs(pos, vel, acc)
        return acc

    def refresh(self):
        """重建 PN 星对索引：质量 ≥ _PN_MIN_MASS 的活星对。

        碎片的 PN 贡献按质量标度（~m_frag·M²）远小于主星对，跳过
        以保帧预算（N 体时星对数 N² 会爆炸）；质量变更/结构事件后
        须调用（merge_pair / disrupt 内部已调用，直接改 masses 的
        上层需显式调用）。
        """
        idx = [k for k in range(len(self.masses))
               if self.masses[k] >= _PN_MIN_MASS]
        self._pn_idx = tuple((idx[a], idx[b])
                             for a in range(len(idx))
                             for b in range(a + 1, len(idx)))

    def _pn_pairs(self, pos, vel, acc):
        """1PN + 2.5PN 星对修正，就地累加到 acc（公式与适用范围见
        _accel 文档）。纯 Python 标量实现：每对 ~50 次标量运算——
        微型 numpy 数组的调度开销远大于计算本身，标量实测快约 37 倍
        （4μs vs 148μs/步）；星对集合由 refresh 维护（质量门槛自动
        排除碎片），死星（质量 0）、同位对（r²<1e-18）与零相对
        速度对自动跳过。
        """
        if not self._pn_idx:
            return
        g = self.G
        g3 = g * g * g
        c2i = _C_INV_LIGHT2
        c5i = _C_INV_LIGHT5
        ms = self.masses.tolist()
        ps = pos.tolist()
        vs = vel.tolist()
        # 每星的累加器（Python 列表，避免微型 numpy 开销）
        n = len(ms)
        accs = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i, j in self._pn_idx:
            mi = ms[i]
            mj = ms[j]
            if mi == 0.0 or mj == 0.0:
                continue
            pi = ps[i]
            pj = ps[j]
            rx = pi[0] - pj[0]
            ry = pi[1] - pj[1]
            rz = pi[2] - pj[2]
            vi = vs[i]
            vj = vs[j]
            vx = vi[0] - vj[0]
            vy = vi[1] - vj[1]
            vz = vi[2] - vj[2]
            r2 = rx * rx + ry * ry + rz * rz
            v2 = vx * vx + vy * vy + vz * vz
            if r2 < 1e-18 or v2 < 1e-30:
                continue
            rn = math.sqrt(r2)
            inv_r = 1.0 / rn
            rdot = (rx * vx + ry * vy + rz * vz) * inv_r   # ṙ = n·v
            Mij = mi + mj
            nu = (mi * mj) / (Mij * Mij)                   # 对称质量比
            GM = g * Mij
            # 1PN（谐和规范，两体精确）：
            #   κ = 2(2+ν)GM/r - (1+3ν)v² + (3/2)νṙ²，切向项 (4-2ν)ṙ·v
            kappa = ((4.0 + 2.0 * nu) * GM * inv_r
                     - (1.0 + 3.0 * nu) * v2
                     + 1.5 * nu * rdot * rdot)
            c1 = GM * c2i * inv_r * inv_r * inv_r
            c2 = (4.0 - 2.0 * nu) * rdot * rn
            ax = c1 * (rx * kappa + c2 * vx)
            ay = c1 * (ry * kappa + c2 * vy)
            az = c1 * (rz * kappa + c2 * vz)
            # 2.5PN：能量平衡拖曳 -λ·v（圆轨严格 Peters，见 _accel 文档）
            lam = (_PN_GW_K * g3 * mi * mj * Mij * c5i / (r2 * r2)
                   * (12.0 * v2 - 11.0 * rdot * rdot) / v2)
            ax -= lam * vx
            ay -= lam * vy
            az -= lam * vz
            # 动量守恒分摊：i 得 (m_j/M)·a_rel，j 得反向 (m_i/M)·a_rel
            wi = mj / Mij
            wj = mi / Mij
            ai = accs[i]
            aj = accs[j]
            ai[0] += wi * ax
            ai[1] += wi * ay
            ai[2] += wi * az
            aj[0] -= wj * ax
            aj[1] -= wj * ay
            aj[2] -= wj * az
        for k in range(n):
            if accs[k][0] != 0.0 or accs[k][1] != 0.0 or accs[k][2] != 0.0:
                acc[k] += accs[k]

    def step(self, dt):
        """一个经典 RK4 步长（状态 (pos, vel) 的一阶系统形式）。

        加速度含速度依赖的 PN 项，须按全状态 RK4 分级（旧版二阶
        特化形式仅在 a(p) 与速度无关时等价四阶）；PN 项为零时与旧
        公式逐位一致。
        """
        p, v = self.pos, self.vel
        k1p = v
        k1v = self._accel(p, v)
        k2p = v + 0.5 * dt * k1v
        k2v = self._accel(p + 0.5 * dt * k1p, k2p)
        k3p = v + 0.5 * dt * k2v
        k3v = self._accel(p + 0.5 * dt * k2p, k3p)
        k4p = v + dt * k3v
        k4v = self._accel(p + dt * k3p, k4p)
        self.pos = p + dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        self.vel = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
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
        self.refresh()

    # ------------------------------------------------------------ 潮汐物理

    def encounter_scan(self, radii):
        """单次距离矩阵联合扫描：表面接触 + 潮汐瓦解事件。

        返回 (contact, tides)：
          contact  首对表面接触的活星 (i, j)（取最近者；无则 None）。
            同批次兄弟碎片在保护期内（_TIDE_GRACE）互撞豁免 —— 刚
            撕裂的碎片仍在相互穿越，靠剪切/膨胀散开后才恢复碰撞
            （回落吸积成团是物理图景的一部分）。
          tides    [(i, j, depth), ...] 潮汐事件（深度降序；i 由 j 施潮，
            depth = d_R/d 为洛希深度）。分级：1 < depth ≤ _TIDE_STRIP_MAX
            为浅区逐层剥离（strip：每次撕下外围 ~20% 成碎屑，母核
            存续并冷却，多次穿越/持续深入中逐层溶解）；depth 更大
            为深区整体瓦解（disrupt：撕成少量大块，大块继续被撕，
            级联逐层细化）。质量低于 _TIDE_FLOOR 的碎片与保护期内
            的天体不再瓦解（防级联指数爆炸）。等密度天体的洛希
            半径化简为 2.2·r_j（扰动星半径）—— 等质量流体星总在
            接触前互相撕裂，大质量比时小星先被撕成流、大星几乎
            不变形，均与真实接触双星/潮汐瓦解事件（TDE）图景一致。
        """
        n = len(self.masses)
        contact = None
        tides = []
        if n < 2:
            return contact, tides
        diff = self.pos[None, :, :] - self.pos[:, None, :]
        dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        np.fill_diagonal(dist, 1e9)
        alive = self.alive
        if alive.sum() >= 2:
            # ---- 接触（含兄弟碎片保护期豁免；违例按距离取最近） ----
            sumr = radii[:, None] + radii[None, :]
            viol = (dist < sumr) & alive[:, None] & alive[None, :]
            iu, ju = np.triu_indices(n, k=1)
            vm = viol[iu, ju]
            if vm.any():
                dm = dist[iu, ju][vm]
                ii, jj = iu[vm], ju[vm]
                for q in np.argsort(dm):
                    a, b = int(ii[q]), int(jj[q])
                    if (self.cohort[a] == self.cohort[b] and self.cohort[a] >= 1
                            and self.time - self.birth[a] < _TIDE_GRACE):
                        continue
                    contact = (a, b)
                    break
        # ---- 潮汐瓦解 ----
        fresh = self.time - self.birth < _TIDE_GRACE
        elig = alive & ~fresh & (self.masses >= _TIDE_FLOOR)
        if elig.any():
            m_safe = np.where(self.masses > 0.0, self.masses, 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (m_safe[None, :] / m_safe[:, None]) ** (1.0 / 3.0)
            dR = _TIDE_Q * radii[:, None] * ratio
            src_ok = alive & (self.masses > 0.0)
            tmask = (dR > dist) & elig[:, None] & src_ok[None, :]
            if tmask.any():
                depth = np.where(tmask, dR / np.maximum(dist, 1e-9), 0.0)
                for i in np.nonzero(elig)[0]:
                    j = int(np.argmax(depth[i]))
                    if depth[i, j] > 1.0:
                        tides.append((int(i), j, float(depth[i, j])))
                tides.sort(key=lambda e: -e[2])
        return contact, tides

    def contact_pair(self, radii):
        """首对表面接触的活星（d < r_i + r_j），无则返回 None。

        （encounter_scan 的接触部分单独暴露，供随机配置预演等
        只关心碰撞的调用方使用。）"""
        return self.encounter_scan(radii)[0]

    def disrupt(self, i, j, radii, max_bodies=MAX_BODIES):
        """深区潮汐瓦解天体 i（施潮体 j）：i 撕裂为少量大块。

        渐进式策略：单次只撕成 2-5 个大块（而非一次性粉碎）——
        大块质量 ≥ 瓦解下限，仍在下落中会被再次撕碎，级联逐层
        细化（配合浅区剥离 strip 与粒子视觉，恒星星现“逐渐溶解”
        而非瞬时消失）。槽位策略：最大块重用槽 i（尾迹无缝延续
        为流的主块），其余追加新槽。返回碎片槽位列表；预算
        （max_bodies）不足返回 None（上层可退化为并合）。

        碎片运动学（真实 TDE 图景的近似）：
          位置：沿潮汐轴（指向施潮体）±1.1·R 紧凑排布 + 横向弥散；
          速度 = 母星速度 + 开普勒剪切 0.5·Ω·x·t̂（近侧加速坠入、
            远侧减速逃逸 -> 沿轨道拉成流）+ 径向解绑膨胀（~0.45·v_esc，
            潮汐势克服自引力做的功）+ 各向弥散（~0.22·v_esc）。
          质量：较均匀的幂律（大块略重），均值 ~m/n —— 大块仍
            可再撕，级联在 2-3 层内自然终止于瓦解下限。
        质量与动量严格守恒（位置/速度按质量加权归心校正）；能量
        近似（注入 ~½·m·v_esc² 量级的弥散动能，物理上来自潮汐势
        与轨道能的交换）。碎片密度 = 母星密度（类型不变；MS 撕成
        MS 流、WD 撕成 WD 流；NS/BH 半径太小，接触并合总在洛希
        界限之内，不会走到这里 —— 但 NS-NS 接触前的潮汐尾仍是
        真实物理，允许发生）。
        """
        n0 = len(self.masses)
        budget = max_bodies - n0 + 1          # 槽 i 重用一个槽位
        m = float(self.masses[i])
        ri = float(radii[i])
        if m <= 0.0 or ri <= 0.0:
            return None
        n_want = int(np.clip(round(m / 0.9), 2, _TIDE_MAXFRAG))
        n_frag = min(n_want, budget)
        if n_frag < 2:
            return None
        rng = np.random.default_rng()
        # ---- 较均匀的幂律质量分布与同密度半径 ----
        w = rng.uniform(0.45, 1.0, n_frag) ** 1.1
        mf = m * w / w.sum()
        rf = ri * (mf / m) ** (1.0 / 3.0)
        # ---- 潮汐轴 n̂（指向施潮体）与轨道切向 t̂ ----
        dvec = self.pos[j] - self.pos[i]
        d = float(np.linalg.norm(dvec))
        if d < 1e-9:
            return None
        nhat = dvec / d
        vrel = self.vel[i] - self.vel[j]
        vperp = vrel - nhat * float(vrel.dot(nhat))
        if float(np.linalg.norm(vperp)) > 1e-6:
            that = vperp / np.linalg.norm(vperp)
        else:                                # 纯径向坠落：取任意正交向
            ref = (np.array([0.0, 0.0, 1.0]) if abs(nhat[2]) < 0.9
                   else np.array([1.0, 0.0, 0.0]))
            that = np.cross(nhat, ref)
            that /= np.linalg.norm(that)
        omega = math.sqrt(self.G * (m + float(self.masses[j])) / d ** 3)
        vesc = math.sqrt(2.0 * self.G * m / ri)
        # ---- 紧凑排布（±1.1R；初始重叠由保护期豁免，之后靠剪切散开） ----
        x = np.linspace(-1.1 * ri, 1.1 * ri, n_frag)
        x += 0.08 * ri * rng.uniform(-1.0, 1.0, n_frag)
        side = rng.normal(0.0, 0.22, (n_frag, 3)) * rf[:, None]
        pnew = self.pos[i] + x[:, None] * nhat + side
        vnew = (self.vel[i] + 0.5 * omega * x[:, None] * that
                + (0.45 * vesc / (1.1 * ri)) * x[:, None] * nhat
                + 0.22 * vesc * rng.normal(0.0, 1.0, (n_frag, 3)))
        # 质心/动量严格守恒：按质量加权归心校正
        pnew -= (mf[:, None] * (pnew - self.pos[i])).sum(0) / m
        vnew -= (mf[:, None] * (vnew - self.vel[i])).sum(0) / m
        # ---- 槽位手术：追加重置零，最大碎片重用槽 i ----
        add = n_frag - 1
        self.masses = np.concatenate([self.masses, np.zeros(add)])
        self.pos = np.vstack([self.pos, np.zeros((add, 3))])
        self.vel = np.vstack([self.vel, np.zeros((add, 3))])
        self.alive = np.concatenate([self.alive, np.ones(add, dtype=bool)])
        self.host = np.concatenate([self.host, np.full(add, -1, np.int64)])
        self.birth = np.concatenate([self.birth, np.full(add, -1e9)])
        self.cohort = np.concatenate([self.cohort, np.zeros(add, np.int64)])
        self._eid += 1
        eid = self._eid
        big = int(np.argmax(mf))
        slots = []
        s = n0
        for q in range(n_frag):
            if q == big:
                k = i
            else:
                k = s
                s += 1
            self.masses[k] = mf[q]
            self.pos[k] = pnew[q]
            self.vel[k] = vnew[q]
            self.birth[k] = self.time
            self.cohort[k] = eid
            slots.append(k)
        self.refresh()
        return slots

    def strip(self, i, j, radii, frac=_TIDE_PEEL, max_bodies=MAX_BODIES):
        """浅区潮汐剥离：从天体 i 撕下外围 ~frac 质量成 2 颗碎屑。

        深入洛希区但未及粉碎带（1 < depth ≤ _TIDE_STRIP_MAX）时的
        事件：恒星先被撕掉外围包层（真实 TDE 的 partial stripping /
        repeated stripping 图景），母核存续、半径随质量缩小，并获得
        与碎片相同的出生冷却（下一层剥离/瓦解在保护期后才会发生，
        呈现逐层溶解而非瞬间消失）。碎屑质量刻意压到瓦解下限以下
        （不可再撕的"颗粒"），沿潮汐轴两侧甩出（近侧坠向施潮体、
        远侧拖尾），带开普勒剪切与径向解绑。

        动量/质心严格守恒（碎屑速度/位置按质量加权归心，母核速度
        不变）；槽位预算不足时返回 None（浅剥离不退化为并合，由
        上层静默跳过 —— 深入后自然走 disrupt/接触路径）。
        """
        n0 = len(self.masses)
        if max_bodies - n0 < 2:
            return None
        m = float(self.masses[i])
        ri = float(radii[i])
        if m <= 0.0 or ri <= 0.0:
            return None
        m_core = max(m * (1.0 - frac), _TIDE_FLOOR)   # 母核保有可再撕的下限
        m_peel = m - m_core
        if m_peel < 0.02 * m:
            return None
        rng = np.random.default_rng()
        dvec = self.pos[j] - self.pos[i]
        d = float(np.linalg.norm(dvec))
        if d < 1e-9:
            return None
        nhat = dvec / d
        vrel = self.vel[i] - self.vel[j]
        vperp = vrel - nhat * float(vrel.dot(nhat))
        if float(np.linalg.norm(vperp)) > 1e-6:
            that = vperp / np.linalg.norm(vperp)
        else:
            ref = (np.array([0.0, 0.0, 1.0]) if abs(nhat[2]) < 0.9
                   else np.array([1.0, 0.0, 0.0]))
            that = np.cross(nhat, ref)
            that /= np.linalg.norm(that)
        omega = math.sqrt(self.G * (m + float(self.masses[j])) / d ** 3)
        vesc = math.sqrt(2.0 * self.G * m / ri)
        # ---- 2 颗碎屑：近侧一颗坠向施潮体、远侧一颗拖尾 ----
        mf = np.array([0.55, 0.45]) * m_peel
        rf = ri * (mf / m) ** (1.0 / 3.0)
        x = np.array([0.95, -0.75]) * ri
        side = rng.normal(0.0, 0.16, (2, 3)) * rf[:, None]
        pnew = self.pos[i] + x[:, None] * nhat + side
        vnew = (self.vel[i] + 0.5 * omega * x[:, None] * that
                + (0.30 * vesc / ri) * x[:, None] * nhat
                + 0.16 * vesc * rng.normal(0.0, 1.0, (2, 3)))
        pnew -= (mf[:, None] * (pnew - self.pos[i])).sum(0) / m
        vnew -= (mf[:, None] * (vnew - self.vel[i])).sum(0) / m
        # ---- 槽位追加 + 母核冷却（与碎屑同批次，互撞豁免） ----
        self.masses = np.concatenate([self.masses, np.zeros(2)])
        self.pos = np.vstack([self.pos, np.zeros((2, 3))])
        self.vel = np.vstack([self.vel, np.zeros((2, 3))])
        self.alive = np.concatenate([self.alive, np.ones(2, dtype=bool)])
        self.host = np.concatenate([self.host, np.full(2, -1, np.int64)])
        self.birth = np.concatenate([self.birth, np.full(2, -1e9)])
        self.cohort = np.concatenate([self.cohort, np.zeros(2, np.int64)])
        self._eid += 1
        eid = self._eid
        slots = []
        for q in range(2):
            k = n0 + q
            self.masses[k] = mf[q]
            self.pos[k] = pnew[q]
            self.vel[k] = vnew[q]
            self.birth[k] = self.time
            self.cohort[k] = eid
            slots.append(k)
        self.masses[i] = m_core       # 母核减重（半径随密度守恒缩小）
        self.birth[i] = self.time
        self.cohort[i] = eid
        self.refresh()
        return slots

    def tidal_state(self, radii):
        """每体的潮汐拉伸诊断（纯视觉，不影响动力学）。

        返回 (stretch, axis)：stretch 为沿潮汐轴的拉伸因子（体积
        守恒：横向自动压缩 1/√f），axis 为潮汐轴单位向量（指向主导
        扰动体）。度规：x = q³·Σ_j (m_j/m_i)(r_i/d_j)³ —— 单一
        扰动源时恰为 (d_R/d)³（洛希半径与间距之比的立方）；
        x = 1 即瓦解阈值。拉伸曲线 f = 1 + 0.8x + 0.5x³（平衡潮
        线性项 + 近瓦解非线性陡增）：x=0.3 时 f≈1.25（贴近可感知），
        x→1 时 f≈2.3（撕裂前的极限拉长）。致密天体（NS/BH）半径
        极小，x 恒近零，天然不变形。
        """
        n = len(self.masses)
        stretch = np.ones(n)
        axis = np.zeros((n, 3))
        axis[:, 2] = 1.0
        if n < 2 or not self.alive.any():
            return stretch, axis
        m_safe = np.where(self.masses > 0.0, self.masses, 1.0)
        diff = self.pos[None, :, :] - self.pos[:, None, :]
        dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        np.fill_diagonal(dist, 1e9)
        ok = self.alive[None, :] & (self.masses[None, :] > 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = _TIDE_Q ** 3 * (m_safe[None, :] / m_safe[:, None]) \
                * (radii[:, None] / dist) ** 3
        w = np.where(ok, w, 0.0)
        # 黑洞（半径恰为史瓦西半径）是真空白：无物质表面，不变形
        is_bh = radii <= 2.0 * self.masses / C_LIGHT2 * 1.001
        w[is_bh, :] = 0.0
        x = w.sum(axis=1)
        for i in range(n):
            if not self.alive[i] or x[i] <= 1e-4:
                continue
            xc = min(x[i], 1.05)
            stretch[i] = 1.0 + 0.8 * xc + 0.5 * xc * xc * xc
            ax = (w[i, :, None] * diff[i, :, :]).sum(0)  # 指向各扰动体
            na = float(np.linalg.norm(ax))
            if na > 1e-12:
                axis[i] = ax / na
        return stretch, axis

    def min_pair_dist(self):
        """最近两活星间距（自适应步长的加密依据；死星幽灵不参与）"""
        p = self.pos[self.alive]
        if len(p) < 2:
            return 1e9
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        iu = np.triu_indices(len(p), k=1)
        return float(np.sqrt(d2[iu].min())) if len(iu[0]) else 1e9

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
        # 1.2~3.6），两颗 m=1.2 主序星对称绕行 —— 欧拉共线构型的
        # 相对平衡解。牛顿值为 v² = G(M_bh+m/4)/r = 1.26（v=1.122）；
        # 1PN 常开下圆轨速度下移（Ω² = GM/r³·[1-(3-ν)GM/(c²r)]），
        # 数值扫描取 v=1.045（70 单位内偏心率 ~2%）。
        # 2.5PN 引力波辐射使双星缓慢旋近（ȧ ≈ -1.1e-3/单位，
        # a=5 -> 并合约千余单位），终将被吞并合。
        name='Black hole + 2 stars',
        masses=[6.0, 1.2, 1.2],
        pos=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]],
        vel=[[0.0, 0.0, 0.0], [0.0, 1.045, 0.0], [0.0, -1.045, 0.0]],
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
