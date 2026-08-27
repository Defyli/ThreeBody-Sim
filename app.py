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
from physics import (CONFIG_ORDER, MAX_BODIES, TYPE_BH, TYPE_NS,
                     ThreeBodySystem, get_config, star_radius, star_type)
from physics import _TIDE_STRIP_MAX
from render import pipeline
from render.context import (BLOOM_THR, C_INV_LIGHT2, EXPOSURE_DEF,
                            FUSE_RANGE, GLOW_OUT_SIG, LENS_CUT,
                            MAX_FUSE, MAX_LENS, MAX_PART, RES, STAR_TINTS,
                            TAIL_MAX, TYPE_GAIN, TYPE_SHORT, TYPE_TINTS)
from render.state import (SI_AXIS, SI_BB, SI_CAM, SI_FOV, SI_FUSEI,
                          SI_FUSEJ, SI_FUSEN, SI_GAIN, SI_LENSK, SI_LENSN,
                          SI_LOOK, SI_NPART, SI_PAL, SI_PCOL, SI_PPOS,
                          SI_PRAD, SI_POS, SI_STR, SI_TCNT, SI_TPRE,
                          SI_TPTS, STAGE_N, n_body_f, star_mass_f,
                          star_pos_f, star_rad_f, star_seeds, star_tints,
                          star_type_f, stage)
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


class ParticlePool:
    """事件粒子池（潮汐撕裂喷发 / 碎片蒸发尾 / 并合溅射）。

    纯视觉粒子：不参与引力与碰撞（短寿命、直线或微减速飞行），
    模拟时间推进（暂停时冻结、倍速时同步加快）。numpy 环形缓冲
    维护（容量 MAX_PART，溢出覆盖最旧）；每帧向量化步进 + 有效
    粒子紧凑打包上传（alpha 为亮度包络：出生快速淡入、老化渐隐、
    末端指数收尾），GPU 端以高斯点精灵 splat 进 HDR（喂 bloom）。
    """

    def __init__(self, cap=MAX_PART):
        self.cap = cap
        self.pos = np.zeros((cap, 3), np.float32)
        self.vel = np.zeros((cap, 3), np.float32)
        self.col = np.zeros((cap, 3), np.float32)
        self.rad = np.zeros(cap, np.float32)
        self.age = np.zeros(cap, np.float32)
        self.life = np.zeros(cap, np.float32)
        self.head = 0
        # 打包缓冲（预分配复用；仅前 n_packed 个有效槽位被 kernel 读取，
        # 尾部残留无需清零 —— 计数截断保证了这一点）
        self._up_pos = np.zeros((cap, 3), np.float32)
        self._up_col = np.zeros((cap, 3), np.float32)
        self._up_rad = np.zeros(cap, np.float32)
        self._up_alpha = np.zeros(cap, np.float32)
        self.n_packed = 0

    def clear(self):
        self.age[:] = self.life[:] + 1.0        # 全部判死

    def spawn(self, pos, vel, col, rad, life):
        """批量发射（向量化）：pos/vel/col 为 (n,3)，rad/life 为 (n,)"""
        n = len(pos)
        if n <= 0:
            return
        idx = (self.head + np.arange(n)) % self.cap
        self.head = int((self.head + n) % self.cap)
        self.pos[idx] = pos
        self.vel[idx] = vel
        self.col[idx] = col
        self.rad[idx] = rad
        self.age[idx] = 0.0
        self.life[idx] = life

    def update(self, dt):
        """模拟时间步进（向量化）：直线飞行 + 微衰减"""
        live = self.age < self.life
        if live.any():
            self.age[live] += dt
            self.pos[live] += self.vel[live] * dt

    def pack(self):
        """紧凑打包有效粒子（含亮度包络 alpha）到复用缓冲。

        有效粒子排在数组前缀，消费 kernel 只读前 n_packed 个槽位
        —— 尾部残留的旧数据无需清零；GPU 上传由上层统一打包进
        staging（见 state.stage）。返回有效粒子数。
        """
        idx = np.nonzero(self.age < self.life)[0]
        n = len(idx)
        self.n_packed = n
        if n == 0:
            return n
        # 亮度包络：前 12% 寿命快速淡入，之后线性衰减、末端再收尾
        tt = self.age[idx] / np.maximum(self.life[idx], 1e-6)
        a = np.clip(tt / 0.12, 0.0, 1.0) * (1.0 - tt) ** 1.5
        self._up_pos[:n] = self.pos[idx]
        self._up_col[:n] = self.col[idx]
        self._up_rad[:n] = self.rad[idx]
        self._up_alpha[:n] = a
        return n


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
        self.collisions = None    # 首次 reset 按配置初始化；此后由用户勾选控制
        self.frame_id = 0
        self.fps = 60.0
        self._fps_t = time.perf_counter()
        self.anim_t = 0.0            # 表面动画时钟（小数值，保证 kernel 内 f32 精度）
        self._last_wall = None
        self._flash = {}             # 星号 -> 并合时刻（临时增亮闪光）
        self._dead_trail = {}        # 死星 -> (末尾点数, 熄灭时刻)，尾迹渐隐用

        # 尾迹环形缓冲（numpy 侧维护，渲染时展开上传）
        self.trails = TrailBuffer()
        # 事件粒子池（潮汐撕裂喷发 / 并合溅射；视觉层，不参与引力）
        self.particles = ParticlePool()
        # 每帧动态输入的 staging 打包缓冲（批量上传通道，见 render）
        self._stage = np.zeros(STAGE_N, np.float32)
        # 空包围盒模式（所有槽位初始化为“无可见贡献”；每帧先铺底再覆写）
        self._bb_empty = np.tile(np.array((1e9, 1e9, -1e9, -1e9),
                                          np.float32), MAX_BODIES)
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
        # 仅首次构造按配置取默认（特解默认关碰撞）；此后切换配置/
        # 重置保留用户勾选 —— 用户的选择优先于配置预设
        if self.collisions is None:
            self.collisions = bool(cfg.get('collide', True))
        self._flash = {}
        self._dead_trail = {}

        self.trails.reset()
        self.particles.clear()      # 新宇宙不继承旧宇宙的粒子

        # 恒星视觉参数（质量 + 密度 -> 半径/类型/色调/增益，统一在
        # _sync_visual 计算；star_tint_np 为基色，致密天体色由类型覆盖；
        # 潮汐瓦解增生碎片时这些数组同步追加，见 _register_debris）
        self.star_tint_np = np.array(STAR_TINTS, dtype=np.float32)
        self.density = np.asarray(cfg.get('density', (3.0, 3.0, 3.0)),
                                  dtype=np.float64).copy()
        self.star_seeds_np = np.array(
            [[0.31, 0.77, 0.19], [0.83, 0.12, 0.55], [0.57, 0.41, 0.92]],
            dtype=np.float32)
        star_pos_f.from_numpy(self._pad2(self.sys.pos))
        self.set_masses(cfg['masses'])
        self.trails.record(self.sys.pos.astype(np.float32),
                           self._trail_alive())   # 需 star_types，在 set_masses 后

        # 相机初始化
        self.cam_rig.snap(self.sys, cfg['cam_dist'])
        self.cam_rig.place_cinematic(0.0, self.sys)
        types = '/'.join(TYPE_SHORT[int(t)] for t in star_type(self.density))
        print(f'[sim3d] 配置: {cfg["name"]}  质量: '
              f'{np.round(np.asarray(cfg["masses"]), 2).tolist()}'
              f'  类型: {types}')

    @staticmethod
    def _pad1(a, fill=0.0):
        """一维数组补齐到 MAX_BODIES 槽位（field 定形上传）"""
        out = np.full(MAX_BODIES, fill, dtype=np.float32)
        out[:len(a)] = a
        return out

    @staticmethod
    def _pad2(a, fill=0.0):
        """二维 (n, 3) 数组补齐到 MAX_BODIES 槽位"""
        out = np.full((MAX_BODIES, 3), fill, dtype=np.float32)
        out[:len(a)] = a
        return out

    @staticmethod
    def _gain_for(tints):
        """按色调亮度归一化增益（红星色调亮度低 → 增益补尝）"""
        lum = (0.2126 * tints[:, 0] + 0.7152 * tints[:, 1]
               + 0.0722 * tints[:, 2])
        return np.clip((0.72 / np.maximum(lum, 0.3)) ** 0.75, 0.85, 1.5)

    def _sync_visual(self):
        """把质量 + 密度同步为渲染视觉参数（半径/类型/色调/增益/质量/种子）。

        类型由密度决定（physics.star_type）；半径由 physics.star_radius
        （MS/WD 等密度球、NS 鈐到 ≥2R_s、BH 即 R_s）。有效色调：MS
        用基色（含并合混合色、碎片微扰色），致密天体用类型色；增益 =
        色调亮度归一 × 类型增益（NS 极亮喂 bloom）。死星质量/半径
        置 0，渲染与碰撞自动跳过。所有 field 按 MAX_BODIES 槽位补齐
        上传（有效槽数经 n_body_f 同步，kernel 作运行时循环上界）。
        """
        n = len(self.sys.masses)
        m = self.sys.masses.copy()
        m[~self.sys.alive] = 0.0
        ty = star_type(self.density)
        # 活星转黑洞（密度滑条拉满/致密并合）：旧光迹停止记录并
        # 登记渐隐（黑洞无光，冻结的尾迹会永久滞留成僵直亮线）
        old = getattr(self, 'star_types', None)
        if old is not None:
            for i in range(min(len(ty), len(old))):
                if self.sys.alive[i] and ty[i] == TYPE_BH \
                        and old[i] != TYPE_BH and i not in self._dead_trail:
                    self._dead_trail[i] = (int(self.trails.count[i]),
                                           self.sys.time)
        self.star_types = ty
        self.star_radius = star_radius(m, self.density)
        self.star_radius[~self.sys.alive] = 0.0   # 死星彻底退出渲染/碰撞
        eff = np.zeros((n, 3), dtype=np.float32)
        nt = min(n, len(self.star_tint_np))
        eff[:nt] = self.star_tint_np[:nt]
        for i in range(n):
            t = int(ty[i])
            if t > 0:
                eff[i] = np.array(TYPE_TINTS[t], dtype=np.float32)
        gain = self._gain_for(eff)
        for i in range(n):
            gain[i] *= TYPE_GAIN[int(ty[i])]
        self.star_gain_np = gain
        # 增益不含闪光（每帧动态）——随 render 的 staging 通道上传；
        # 其余为结构级静态数据，仅在结构变化时低频直传
        n_body_f.from_numpy(np.array([n], dtype=np.int32))
        star_rad_f.from_numpy(self._pad1(self.star_radius))
        star_type_f.from_numpy(self._pad1(ty.astype(np.int32)))
        star_mass_f.from_numpy(self._pad1(m))
        star_tints.from_numpy(self._pad2(eff))
        star_seeds.from_numpy(self._pad2(self.star_seeds_np[:n]))

    def set_masses(self, masses):
        """运行时修改天体质量（不重置轨迹）：同步物理质量与视觉参数。

        质量直接改变引力，轨道会随之演化（这正是“混沌玩法”的乐趣）；
        半径/类型/色调/增益由 _sync_visual 按质量 + 密度统一计算。
        已并合熄灭的星质量恒为 0（半径也为 0，渲染自动跳过）。质量
        变更后重建 PN 星对索引（碎片质量门槛判定，见 sys.refresh）。
        """
        m = np.asarray(masses, dtype=np.float64).copy()
        m[~self.sys.alive] = 0.0
        self.sys.masses = m
        self.sys.refresh()
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
        """单个物理步：积分 + 潮汐瓦解 + 碰撞检测 + 尾迹采样"""
        step_fn = (self.sys.step_adaptive if self.cfg.get('adaptive')
                   else self.sys.step)
        step_fn(self.dt)
        if self.collisions:
            contact, tides = self.sys.encounter_scan(self.star_radius)
            if tides:
                self._apply_disruptions(tides)
                # 结构已变（碎片增生/并合），重扫接触
                contact, _ = self.sys.encounter_scan(self.star_radius)
            if contact is not None:
                self._apply_merge(*contact)
        self.trails.step(self.dt, self.sys.pos.astype(np.float32),
                         self._trail_alive())
        self.particles.update(self.dt)   # 事件粒子随模拟时间飞行

    def _apply_disruptions(self, tides):
        """潮汐瓦解事件批次处理（同一扫描内可含互撕双方）。

        洛希深度分级：浅区（1 < depth ≤ _TIDE_STRIP_MAX）逐层剥离
        外围包层（strip：母核存续、半径逐次缩小，伴喷流粒子 ——
        反复穿越中逐层溶解）；深区整体瓦解（disrupt：撕成少量大块
        + 剧烈喷发粒子，大块继续被撕级联细化）。槽位预算耗尽时深
        区退化为就地并合（已深入洛希区，并合在即），浅区静默跳过
        （深入后自然走 disrupt/接触路径）。碎片密度/色调/种子在
        _register_debris 注册；黑洞是真空解不撕裂。
        """
        for i, j, depth in tides:
            if self.star_types[i] == TYPE_BH:
                continue
            if depth <= _TIDE_STRIP_MAX:
                slots = self.sys.strip(i, j, self.star_radius)
                if slots is not None:
                    self._register_debris(i, slots, '潮汐剥离')
                    self._flash[i] = self.sys.time   # 母核短促增亮（剥层节奏）
                    self._emit_strip(i, j)
            else:
                slots = self.sys.disrupt(i, j, self.star_radius)
                if slots is None:
                    self._apply_merge(i, j)
                    return
                self._register_debris(i, slots)
                self._emit_disrupt(i, slots)
        self._sync_visual()      # 重算碎片半径/类型/增益

    def _register_debris(self, parent, slots, verb='潮汐瓦解'):
        """碎片视觉参数注册：密度继承母星（类型不变，MS 撕成 MS 流、
        WD 撕成 WD 流），色调为母色微扰（流的整体色一致），种子随机。
        撕裂瞬间全部碎片短暂增亮（闪光喂 bloom，呈现爆发感）。"""
        add = len(self.sys.masses) - len(self.density)
        if add > 0:
            rho = float(self.density[parent])
            rng = np.random.default_rng()
            self.density = np.concatenate([self.density, np.full(add, rho)])
            tint = self.star_tint_np[parent]
            newt = np.clip(tint[None, :] * (0.88 + 0.24 * rng.random((add, 3))),
                           0.05, 1.6)
            self.star_tint_np = np.vstack(
                [self.star_tint_np, newt.astype(np.float32)])
            self.star_seeds_np = np.vstack(
                [self.star_seeds_np, rng.random((add, 3)).astype(np.float32)])
        for s in slots:
            self._flash[s] = self.sys.time
        print(f'[sim3d] t={self.sys.time:7.2f}  恒星{parent + 1} {verb}'
              f' -> {len(slots)} 碎片  余体{int(self.sys.alive.sum())}')

    # ------------------------------------------------------------ 粒子事件

    def _emit_disrupt(self, parent, slots):
        """深区瓦解喷发：炽热物质云随碎片四微的粒子爆发。

        粒子生成在碎片周围（继承碎片速度 + 各向弥散，速度尺度取母
        星逃逸速度的分散 —— 与碎片的解绑能同源）；色为母色调增亮
        （摩擦加热的白热偏置）。纯视觉层，不参与引力/碰撞。
        """
        pp = self.sys.pos[slots].astype(np.float64)
        vv = self.sys.vel[slots].astype(np.float64)
        tint = self.star_tint_np[parent].astype(np.float64)
        r0 = float(self.star_radius[parent])           # 母星原半径（未同步）
        m0 = float(self.sys.masses[slots].sum())       # 母星质量（分散于碎片）
        vesc = math.sqrt(2.0 * self.sys.G * max(m0, 1e-6)
                         / max(r0, 1e-6))
        nf = len(slots)
        n = int(min(110 * nf, 460))
        rng = np.random.default_rng()
        rep = rng.integers(0, nf, n)                   # 各碎片均沾
        d = rng.normal(size=(n, 3))
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        off = rng.normal(0.0, 0.55, (n, 3)) * r0       # 碎片周围簇内弥散
        spd = rng.uniform(0.25, 0.70, n) * vesc
        pos = pp[rep] + off
        vel = vv[rep] + d * spd[:, None]
        col = np.clip(tint[None, :] * rng.uniform(1.15, 1.9, (n, 1))
                      + 0.12, 0.1, 2.2)
        rad = r0 * rng.uniform(0.045, 0.15, n)
        life = rng.uniform(1.2, 3.2, n)
        self.particles.spawn(pos.astype(np.float32), vel.astype(np.float32),
                             col.astype(np.float32), rad.astype(np.float32),
                             life.astype(np.float32))

    def _emit_strip(self, i, j):
        """浅区剥离喷流：沿潮汐轴两侧用出的包层物质粒子。

        剥离发生在指向施潮体的轴两端（近侧坠入、远侧拖尾），粒子
        速度沿轴 ± 偏置；规模显著小于深区瓦解（表层物质少）。
        """
        p0 = self.sys.pos[i].astype(np.float64)
        v0 = self.sys.vel[i].astype(np.float64)
        dvec = self.sys.pos[j].astype(np.float64) - p0
        d = float(np.linalg.norm(dvec))
        if d < 1e-9:
            return
        nhat = dvec / d
        r0 = float(self.star_radius[i])
        vesc = math.sqrt(2.0 * self.sys.G
                         * max(float(self.sys.masses[i]), 1e-6)
                         / max(r0, 1e-6))
        n = 150
        rng = np.random.default_rng()
        d3 = rng.normal(size=(n, 3))
        d3 /= np.maximum(np.linalg.norm(d3, axis=1, keepdims=True), 1e-9)
        # 潮汐轴 ± 偏置后重归一（喷流锥而非全向）
        d3 = d3 * 0.62 + nhat[None, :] \
            * rng.choice((-1.0, 1.0), n)[:, None] * 0.52
        d3 /= np.maximum(np.linalg.norm(d3, axis=1, keepdims=True), 1e-9)
        pos = p0[None, :] + d3 * (r0 * rng.uniform(0.55, 1.35, n))[:, None]
        spd = rng.uniform(0.30, 0.75, n) * vesc
        vel = v0[None, :] + d3 * spd[:, None]
        tint = self.star_tint_np[i].astype(np.float64)
        col = np.clip(tint[None, :] * rng.uniform(1.1, 1.8, (n, 1))
                      + 0.10, 0.1, 2.2)
        rad = r0 * rng.uniform(0.04, 0.13, n)
        life = rng.uniform(0.9, 2.4, n)
        self.particles.spawn(pos.astype(np.float32), vel.astype(np.float32),
                             col.astype(np.float32), rad.astype(np.float32),
                             life.astype(np.float32))

    def _emit_merge(self, center, vel, span, tint, v_imp):
        """并合溅射：接触点径向抛出的白热火花（nova 式）。

        速度尺度取碰撞相对速度（动能耗散的直观代理）；色为双方
        混合色大幅白热化。粒度小于撕裂事件（碰撞闪光已由 _flash
        + bloom 呈现主体，粒子是飞溅的“火花”层）。
        """
        n = 220
        rng = np.random.default_rng()
        d = rng.normal(size=(n, 3))
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        pos = center[None, :] + d \
            * (0.45 * span * rng.uniform(0.2, 1.0, n))[:, None]
        spd = rng.uniform(0.25, 0.85, n) * max(0.35 * v_imp, 0.3)
        vel = vel[None, :] + d * spd[:, None]
        col = np.clip(0.62 * tint[None, :] + 0.38, 0.15, 2.4) \
            * rng.uniform(1.0, 1.6, (n, 1))
        rad = max(span, 0.05) * rng.uniform(0.035, 0.11, n)
        life = rng.uniform(0.7, 2.0, n)
        self.particles.spawn(pos.astype(np.float32), vel.astype(np.float32),
                             col.astype(np.float32), rad.astype(np.float32),
                             life.astype(np.float32))

    def _apply_merge(self, i, j):
        """天体 i 与 j 接触并合：动量守恒地合为一星，j 熄灭。

        视觉同步：色调按质量加权混合、增益按混合色调重算、半径按
        新质量重算；死星尾迹停止记录并随模拟时间从旧端渐隐；
        并合星短暂闪光（碰撞动能耗散为光，近似 nova 式增亮）。
        潮汐碎片的回落吸积同样经此路径成团（吞并方增重）。
        """
        mi, mj = self.sys.masses[i], self.sys.masses[j]
        m = mi + mj
        # 并合前接触点/质心速度/碰撞速度（merge_pair 后即被改写）
        pc = 0.5 * (self.sys.pos[i] + self.sys.pos[j])
        vc = (mi * self.sys.vel[i] + mj * self.sys.vel[j]) / m
        v_imp = float(np.linalg.norm(self.sys.vel[i] - self.sys.vel[j]))
        span = float(self.star_radius[i] + self.star_radius[j])
        tint_mix = (mi * self.star_tint_np[i]
                    + mj * self.star_tint_np[j]) / m
        self.sys.merge_pair(i, j)
        self._emit_merge(pc, vc, span, tint_mix, v_imp)

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
              f' 并合  M={m:.2f} -> {ty}  余体{int(self.sys.alive.sum())}')

    def _trail_trim(self):
        """死星尾迹的渐隐修剪量（从最旧端逐点丢弃）"""
        trim = np.zeros(MAX_BODIES, dtype=np.int64)
        for k, (cnt0, t0) in self._dead_trail.items():
            if k < MAX_BODIES:
                trim[k] = int(min(cnt0, cnt0 * (self.sys.time - t0) / 6.0))
        return trim

    def advance(self, wall_dt):
        """按时间倍率推进物理，并维护尾迹环形缓冲

        每帧基准模拟时长固定（0.06 模拟秒 @ speed=1）：步数按配置
        dt 归一。含近碰撞的特解（cfg['adaptive']）用 step_adaptive
        在近距时自动加密子步（远距用满步长，速度与精度兼得）。
        碰撞/潮汐瓦解（可开关）在每个物理步后检测：表面接触即动量
        守恒并合，越过洛希极限则撕裂为碎片流（详见 _apply_merge /
        _apply_disruptions）。
        """
        if self.paused:
            return
        steps = int(round(60.0 * self.speed * 0.001 / self.dt))
        steps = max(1, min(steps, 1200))
        for _ in range(steps):
            self._step_once()

    # ------------------------------------------------------------------ draw

    def _pack_fuse(self, st):
        """筛选近距离星对打包（接触融合辉光的逐对扫描集）。

        只有两星距离 < FUSE_RANGE×半径和时才有可见贡献（激活钟形
        宽度 ~1），N 体时真正贴近的星对寥寥无几 —— 每帧 CPU 端
        O(N²) 扫描（~μs 级）换 kernel 端免于逐像素 O(N²)。"""
        n = len(self.sys.masses)
        r, p, alive = self.star_radius, self.sys.pos, self.sys.alive
        ia = np.zeros(MAX_FUSE, np.float32)   # 尾部填充不参与（fuse_n 截断）
        ib = np.zeros(MAX_FUSE, np.float32)
        k = 0
        for i in range(n):
            if not alive[i] or r[i] <= 0.01:
                continue
            for j in range(i + 1, n):
                if not alive[j] or r[j] <= 0.01:
                    continue
                if float(np.linalg.norm(p[i] - p[j])) \
                        < FUSE_RANGE * (r[i] + r[j]):
                    ia[k] = i
                    ib[k] = j
                    k += 1
                    if k >= MAX_FUSE:
                        break
            if k >= MAX_FUSE:
                break
        st[SI_FUSEN] = k
        st[SI_FUSEI:SI_FUSEI + MAX_FUSE] = ia
        st[SI_FUSEJ:SI_FUSEJ + MAX_FUSE] = ib

    def _pack_lens(self, st):
        """致密天体（NS/BH）紧凑列表打包，返回透镜总开关（0/1）。

        N 体碎片化后绝大多数是 MS 碎块，与透镜无关 —— 透镜/阴影
        kernel 只扫这张短表（典型 0-2 个成员，免逐像素全槽位扫描）；
        超容量的多余致密体降级为直射渲染（碎片流中几乎不出现）。
        列表同时缓存到 self._lens_idx，供 _pack_scr_bb 的环像盒
        扩展使用（两处必须一致：降级体不产生透镜弯曲，也无环像）。
        """
        ty = self.star_types
        lk = np.nonzero(self.sys.alive[:len(ty)]
                        & ((ty == TYPE_NS) | (ty == TYPE_BH)))[0]
        lk = lk[:MAX_LENS]
        self._lens_idx = lk
        st[SI_LENSN] = len(lk)
        kk = np.zeros(MAX_LENS, np.float32)
        kk[:len(lk)] = lk
        st[SI_LENSK:SI_LENSK + MAX_LENS] = kk
        return 1 if len(lk) else 0

    # 包围盒辉光截断半径（外晕 σ 的倍数）。标定按 gamma 编码后的
    # 可见性：外晕线性贡献 AMP·exp(-ρ²)·gain 在截断处须 < 2e-6 ——
    # composite 的 gamma 1/2.2 会把暗部放大（1e-3 线性 ≈ 10/255
    # 可见台阶，即 NS 光晕方形轮廓 bug 的根源之一），2e-6 经 gamma
    # 后 ≈ 0.6/255，叠加背景后不可见。NS（TYPE_GAIN=3.2）为最坏
    # 情况：ρ = √ln(0.14·2.8/2e-6) ≈ 3.5；内晕/日珥衰减更快被覆盖
    _BB_GLOW = 3.5 * GLOW_OUT_SIG
    # 环像盒安全系数：θ_E 用前向距离近似角直径距离、像素换算用
    # 小角近似、主像位移取上界，各引入 <15% 误差 —— 统一 30% 裕量吸收
    _BB_RING = 1.3

    def _pack_scr_bb(self, st, cam, look, stretch):
        """每体屏幕包围盒打包（kernel 逐体早退；见 effects._corona_glow）。

        盒 = 拉伸盘投影 ∪ 辉光截断范围 ∪ 透镜环像区：
        - 辉光截断半径按 gamma 编码后不可见标定（见 _BB_GLOW）；
        - 透镜环像：弱偏折像素沿弯曲后的 rdb 求交/采辉光，而盒按
          未偏折的直射几何预投影 —— 爱因斯坦环/副像出现在透镜
          投影周围（≤1.7θ_E）、主像自直射位置位移 ≤ θ_E²/β，若不
          并入盒则环像沿方形盒缘截断（NS 光晕方形轮廓的另一根源）。
          点透镜 θ_E=√(2R_s·D_ls/(D_L·D_s))；像尺寸 ≤ 源角尺寸
          （主/副像角导数 ≤1）。BH 的视界球也参与求交，同样扩展。
        投影与 pipeline._project 逐位一致（针孔模型、y 轴向上、
        像素中心 +0.5 由 kernel 侧保证）；半径用切线角 asin(wr/z)
        的精确换算而非小角近似 —— 近距掠过时角半径显著大于线性
        估计；相机进入截断球内时退化为全屏盒（正确性优先）。
        """
        W, H = pipeline.IMG_W, pipeline.IMG_H
        cam = np.asarray(cam, np.float64)
        fwd = np.asarray(look, np.float64) - cam
        fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
        right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(fwd, right)
        tanh = math.tan(math.radians(self.cam_rig.fov_deg) * 0.5)
        aspect = W / H
        half_h = 0.5 * H / tanh          # 角度 -> 像素（半屏高）
        # 空盒（x0 > x1）表示该体无可见贡献，kernel 直接跳过
        st[SI_BB:SI_BB + 4 * MAX_BODIES] = self._bb_empty
        n = min(len(self.sys.masses), MAX_BODIES, len(self.star_types))
        # 第一遍：直射盒（盘 ∪ 辉光截断），记录投影几何供环像扩展
        px = np.full(n, 1e9)
        py = np.full(n, 1e9)
        zf = np.zeros(n)               # 前向距离（0 = 背后/无效）
        rad_px = np.zeros(n)           # 直射盒像素半径（含辉光）
        for k in range(n):
            if not self.sys.alive[k]:
                continue
            r = float(self.star_radius[k])
            if r <= 0.01:
                continue
            wr = r if self.star_types[k] == TYPE_BH \
                else max(r * float(stretch[k]), self._BB_GLOW * r)
            v = self.sys.pos[k].astype(np.float64) - cam
            z = float(v.dot(fwd))
            if z <= 0.02:
                continue                     # 相机背后
            cx = (float(v.dot(right)) / (z * tanh * aspect) + 1.0) * 0.5 * W
            cy = (float(v.dot(up)) / (z * tanh) + 1.0) * 0.5 * H
            px[k], py[k], zf[k] = cx, cy, z
            if wr >= z:
                # 相机在截断球内（正确性优先：全屏盒）。
                # rad_px=∞ 使其作为源时环像并集不变；zf 已记录，
                # 作为透镜时环像扩展仍有效（球内弯曲更强而非更弱）
                st[SI_BB + 4 * k:SI_BB + 4 * k + 4] = (0.0, 0.0,
                                                       float(W), float(H))
                rad_px[k] = 1e9
                continue
            zz = math.sqrt(z * z - wr * wr)  # 切线角半径（精确）
            rp = wr / zz / tanh * (0.5 * H)
            rad_px[k] = rp
            st[SI_BB + 4 * k:SI_BB + 4 * k + 4] = (cx - rp, cy - rp,
                                                   cx + rp, cy + rp)
        # 第二遍：透镜环像扩展（源在透镜后方才有弯曲像）。
        # BH 的视界球同样参与 _ray_stars 求交（有表面渲染），其
        # 弯曲像（弧像）与其他源一样需要环像盒
        masses = self.sys.masses
        for L in getattr(self, '_lens_idx', ()):
            L = int(L)
            if zf[L] <= 0.02:
                continue                     # 透镜在相机背后：无弯曲
            rs = 2.0 * float(masses[L]) * C_INV_LIGHT2
            zl = zf[L]
            # 透镜自身影响球：弱偏折区（8-20 R_s）的射线被透镜引力
            # 拉向透镜自身、命中其视界球/盘/辉光 —— 透镜的直射盒按
            # 直线几何太小（BH 仅 R_s 投影），须扩到影响球投影（与
            # render_scene 中 bh_front 的 2.5×LENS_CUT×R_s 一致）。
            # NS 的辉光截断盒（3.5σ_glow）通常已 > 影响球，此处冗余
            inf_r = 2.5 * LENS_CUT * rs
            b = SI_BB + 4 * L
            x0, y0, x1, y1 = st[b], st[b + 1], st[b + 2], st[b + 3]
            if inf_r >= zl:
                # 相机在影响球内：全屏盒
                x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
            else:
                ang = math.asin(inf_r / zl)        # 切线角（精确）
                rp = self._BB_RING * (ang / tanh * (0.5 * H)
                                      + rad_px[L])
                x0 = min(x0, px[L] - rp)
                y0 = min(y0, py[L] - rp)
                x1 = max(x1, px[L] + rp)
                y1 = max(y1, py[L] + rp)
            st[b:b + 4] = (x0, y0, x1, y1)
            for k in range(n):
                if zf[k] <= zl or k == L or rad_px[k] <= 0.0:
                    continue
                zk = zf[k]
                d_ls = float(np.linalg.norm(self.sys.pos[k]
                                            - self.sys.pos[L]))
                if d_ls < 1e-6:
                    continue
                th_e = math.sqrt(2.0 * rs * d_ls / (zl * zk))  # 爱因斯坦角
                th_px = th_e * half_h
                b = SI_BB + 4 * k
                x0, y0, x1, y1 = st[b], st[b + 1], st[b + 2], st[b + 3]
                # 副像/近轴主像带：透镜投影周围 1.7θ_E + 源角尺寸
                rr = self._BB_RING * (1.7 * th_px + rad_px[k])
                x0 = min(x0, px[L] - rr)
                y0 = min(y0, py[L] - rr)
                x1 = max(x1, px[L] + rr)
                y1 = max(y1, py[L] + rr)
                # 主像位移：自直射位置偏移 ≤ θ_E²/β（β 大时贴直射像）
                beta = math.hypot(px[k] - px[L], py[k] - py[L])
                sh = self._BB_RING * th_px * th_px / max(beta, th_px, 1.0)
                x0 = min(x0, px[k] - rad_px[k] - sh)
                y0 = min(y0, py[k] - rad_px[k] - sh)
                x1 = max(x1, px[k] + rad_px[k] + sh)
                y1 = max(y1, py[k] + rad_px[k] + sh)
                st[b:b + 4] = (x0, y0, x1, y1)

    def render(self, wall_t):
        """完整渲染管线：staging 打包上传 -> 场景 -> 尾迹 -> bloom -> 呈现。

        性能：每帧动态数据（相机/恒星/包围盒/星对/尾迹/粒子）统一
        打包进单一 staging 缓冲（state.stage），一次 from_numpy +
        GPU scatter kernel 分发到各 field —— Metal 上每次 from_numpy
        有 ~0.22ms 固定开销，逐 field 直传 ~20 次要 ~4.6ms，批量
        通道 <1ms。结构级静态数据（半径/类型/色调/种子/质量）随
        _sync_visual 低频直传。
        """
        st = self._stage
        n = len(self.sys.masses)
        cam_pos = np.asarray(self.cam_rig.cam.curr_position, dtype=np.float32)
        cam_look = np.asarray(self.cam_rig.cam.curr_lookat, dtype=np.float32)
        st[SI_CAM:SI_CAM + 3] = cam_pos
        st[SI_LOOK:SI_LOOK + 3] = cam_look
        st[SI_FOV] = self.cam_rig.fov_deg
        # 恒星动态：位置 / 潮汐拉伸（撕裂前的拉长可视化）/ 增益
        # （基础值 + 并合/瓦解闪光指数衰减，喂 bloom 呈现爆发）
        stretch, axis = self.sys.tidal_state(self.star_radius)
        gain = self.star_gain_np.copy()
        for k, t0 in self._flash.items():
            age = self.sys.time - t0
            if age >= 0.0:
                gain[k] *= 1.0 + 1.7 * math.exp(-age / 0.8)
        st[SI_GAIN:SI_GAIN + n] = gain
        st[SI_STR:SI_STR + n] = stretch
        st[SI_POS:SI_POS + 3 * n] = self.sys.pos.reshape(-1)
        st[SI_AXIS:SI_AXIS + 3 * n] = axis.reshape(-1)
        # 近距星对 / 致密天体列表 / 屏幕包围盒 / 尾迹 / 事件粒子
        self._pack_fuse(st)
        lens_on = self._pack_lens(st)
        self._pack_scr_bb(st, cam_pos, cam_look, stretch)
        self.trails.pack(self.tail_len, self._trail_trim())
        st[SI_TCNT:SI_TCNT + MAX_BODIES] = self.trails.cnt
        st[SI_TPRE:SI_TPRE + MAX_BODIES + 1] = self.trails.prefix
        st[SI_TPTS:] = self.trails.pts.reshape(-1)
        self.particles.pack()
        st[SI_NPART] = self.particles.n_packed
        st[SI_PRAD:SI_PRAD + MAX_PART] = self.particles._up_rad
        st[SI_PAL:SI_PAL + MAX_PART] = self.particles._up_alpha
        st[SI_PPOS:SI_PPOS + 3 * MAX_PART] = self.particles._up_pos.reshape(-1)
        st[SI_PCOL:SI_PCOL + 3 * MAX_PART] = self.particles._up_col.reshape(-1)
        # 一次批量上传 + GPU 分发（布局见 state.SI_*）
        stage.from_numpy(st)
        pipeline.scatter_stage()

        pipeline.render_scene(self.anim_t, lens_on)
        pipeline.splat_trails()
        pipeline.splat_particles()
        pipeline.bloom_down(BLOOM_THR)
        pipeline.bloom_blur_h()
        pipeline.bloom_blur_v()
        pipeline.composite(self.exposure, self.bloom_str)
        pipeline.copy_to_texture(pipeline.img_tex)
        self.canvas.set_image(pipeline.img_tex)

    # ------------------------------------------------------------------- gui

    # 面板内容高度标定（px；imgui 实测：标题栏 30 / 文本行距 17 /
    # 按钮与滑条行距 23，另加安全余量；改面板内容后需同步更新）
    _GUI_FULL_H = (396.0, 254.0, 370.0)   # 全内容所需面板高度
    _GUI_COMP_H = (296.0, 254.0, 330.0)   # 紧凑内容（省略空行/提示文本）所需高度
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
            n_slot = len(self.sys.masses)
            if n_alive < n_slot or n_slot > 3:
                g.text(f'Bodies : {n_alive} alive'
                       f'  ({n_slot - n_alive} merged/absorbed)')
            self.collisions = g.checkbox('Collisions & tidal breakup',
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
            for i in range(min(3, len(newm))):
                if self.sys.alive[i]:
                    newm[i] = g.slider_float(f'Mass {i + 1}',
                                             float(newm[i]), 0.5, 10.0)
                else:
                    g.text(f'Mass {i + 1}: merged into'
                           f' {self.sys.host[i] + 1}')
            if len(newm) > 3:
                g.text(f'Debris : {int(self.sys.alive[3:].sum())} alive /'
                       f' {len(newm) - 3} spawned')
            if not np.allclose(newm, self.sys.masses, rtol=0.0, atol=1e-4):
                self.set_masses(newm)
            if g.button('Reset masses  [M]'):
                if self.sys.alive.all() and len(self.sys.masses) == 3:
                    self.set_masses(self.cfg['masses'])

            g.text('Densities (MS/WD/NS/BH):')
            for i in range(min(3, len(self.density))):
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
        if not self.show_window:
            return                     # 离屏模式无事件源（get_events 需窗口）
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
                if self.sys.alive.all() and len(self.sys.masses) == 3:
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
