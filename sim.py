# 使用 conda activate lfy 环境
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.rcParams['font.family'] = 'Arial Unicode MS'

class ThreeBodySystem:
    def __init__(self, masses, initial_positions, initial_velocities, G=1.0):
        """
        初始化三体系统
        
        参数:
        masses: 三个天体的质量 [m1, m2, m3]
        initial_positions: 初始位置 [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
        initial_velocities: 初始速度 [[vx1,vy1,vz1], [vx2,vy2,vz2], [vx3,vy3,vz3]]
        G: 引力常数
        """
        self.masses = np.array(masses, dtype=np.float64)
        self.G = G
        self.n_bodies = 3
        
        # 初始状态: 每个天体有6个状态变量 (x,y,z,vx,vy,vz)
        self.state = np.zeros(6 * self.n_bodies)
        
        # 设置初始位置和速度
        for i in range(self.n_bodies):
            self.state[6*i:6*i+3] = initial_positions[i]
            self.state[6*i+3:6*i+6] = initial_velocities[i]
    
    def acceleration(self, state):
        """计算每个天体的加速度"""
        acc = np.zeros_like(state)
        
        # 提取位置和速度
        state_reshaped = state.reshape(self.n_bodies, 6)
        positions = state_reshaped[:, 0:3]
        velocities = state_reshaped[:, 3:6]
        
        # 计算每个天体受到的引力
        for i in range(self.n_bodies):
            total_acc = np.zeros(3)
            
            for j in range(self.n_bodies):
                if i != j:
                    # 计算相对位置向量
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec)
                    
                    # 避免除以零
                    if r > 0:
                        # 引力加速度: a = G * m_j * r_vec / r^3
                        total_acc += self.G * self.masses[j] * r_vec / (r**3)
            
            # 更新加速度
            acc[6*i:6*i+3] = velocities[i]  # 位置导数是速度
            acc[6*i+3:6*i+6] = total_acc    # 速度导数是加速度
        
        return acc
    
    def rk4_step(self, dt):
        """执行一个RK4时间步"""
        k1 = self.acceleration(self.state)
        k2 = self.acceleration(self.state + 0.5 * dt * k1)
        k3 = self.acceleration(self.state + 0.5 * dt * k2)
        k4 = self.acceleration(self.state + dt * k3)
        
        self.state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def get_positions(self):
        """获取当前所有天体的位置"""
        positions = []
        for i in range(self.n_bodies):
            pos = self.state[6*i:6*i+3]
            positions.append(pos.copy())
        return np.array(positions)
    
    def get_velocities(self):
        """获取当前所有天体的速度"""
        velocities = []
        for i in range(self.n_bodies):
            vel = self.state[6*i+3:6*i+6]
            velocities.append(vel.copy())
        return np.array(velocities)

def simulate_three_body(dt=0.01, total_time=100, save_interval=10):
    """
    模拟三体系统
    
    参数:
    dt: 时间步长
    total_time: 总模拟时间
    save_interval: 保存间隔（每多少步输出一次）
    """
    # 示例：使用著名的“毕达哥拉斯三体问题”(Pythagorean three-body problem)
    # 这是一个经典的混沌系统，初始状态静止，但质量和位置不对称
    masses = [3.0, 4.0, 5.0]  # 质量分别为3, 4, 5
    
    # 初始位置：形成一个直角三角形 (边长为3, 4, 5)
    initial_positions = [
        [1.0, 3.0, 0.0],   # 质量3的天体
        [-2.0, -1.0, 0.0], # 质量4的天体
        [1.0, -1.0, 0.0]   # 质量5的天体
    ]
    
    # 初始速度：全部静止
    initial_velocities = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    
    # 创建三体系统
    system = ThreeBodySystem(masses, initial_positions, initial_velocities)
    
    # 存储轨迹
    trajectories = [[] for _ in range(3)]
    time_points = []
    
    # 模拟循环
    n_steps = int(total_time / dt)
    
    print("开始三体系统模拟...")
    print(f"时间步长: {dt}, 总步数: {n_steps}")
    print("-" * 50)
    
    for step in range(n_steps):
        current_time = step * dt
        
        # 每save_interval步输出一次
        if step % save_interval == 0:
            positions = system.get_positions()
            print(f"时间: {current_time:.2f}")
            for i, pos in enumerate(positions):
                print(f"  天体{i+1}: 位置({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print("-" * 30)
        
        # 保存轨迹
        if step % 10 == 0:  # 为了减少数据量，每10步保存一次
            positions = system.get_positions()
            for i in range(3):
                trajectories[i].append(positions[i].copy())
            time_points.append(current_time)
        
        # 执行一个时间步
        system.rk4_step(dt)
    
    return system, trajectories, time_points

def plot_trajectories(trajectories, time_points):
    """绘制三体轨迹"""
    trajectories = np.array(trajectories)  # 形状: (3, n_points, 3)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(131, projection='3d')
    colors = ['r', 'g', 'b']
    labels = ['天体1', '天体2', '天体3']
    
    for i in range(3):
        traj = trajectories[i]
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], label=labels[i], alpha=0.7, linewidth=1)
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                   color=colors[i], s=50, marker='o')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('三体系统3D轨迹')
    ax1.legend()
    ax1.grid(True)
    
    # 2. XY平面投影
    ax2 = fig.add_subplot(132)
    for i in range(3):
        traj = trajectories[i]
        ax2.plot(traj[:, 0], traj[:, 1], 
                color=colors[i], label=labels[i], alpha=0.7, linewidth=1)
        ax2.scatter(traj[-1, 0], traj[-1, 1], 
                   color=colors[i], s=50, marker='o')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY平面投影')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. 时间-距离图
    ax3 = fig.add_subplot(133)
    
    # 计算天体间的距离
    distances_12 = []
    distances_13 = []
    distances_23 = []
    
    for t in range(len(time_points)):
        pos1 = trajectories[0][t]
        pos2 = trajectories[1][t]
        pos3 = trajectories[2][t]
        
        distances_12.append(np.linalg.norm(pos1 - pos2))
        distances_13.append(np.linalg.norm(pos1 - pos3))
        distances_23.append(np.linalg.norm(pos2 - pos3))
    
    ax3.plot(time_points, distances_12, 'r-', label='天体1-2距离', alpha=0.7)
    ax3.plot(time_points, distances_13, 'g-', label='天体1-3距离', alpha=0.7)
    ax3.plot(time_points, distances_23, 'b-', label='天体2-3距离', alpha=0.7)
    
    ax3.set_xlabel('时间')
    ax3.set_ylabel('距离')
    ax3.set_title('天体间距离随时间变化')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def animate_trajectories(trajectories, time_points):
    """动态可视化三体运动过程"""
    trajectories = np.array(trajectories)  # 形状: (3, n_points, 3)
    n_points = len(time_points)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b']
    labels = ['天体1', '天体2', '天体3']
    
    # 初始化轨迹线和当前位置点
    lines = [ax.plot([], [], [], color=colors[i], label=labels[i], alpha=0.7, linewidth=1.5)[0] for i in range(3)]
    points = [ax.plot([], [], [], marker='o', color=colors[i], markersize=8)[0] for i in range(3)]
    
    # 计算坐标轴范围，保持比例一致
    all_x = trajectories[:, :, 0].flatten()
    all_y = trajectories[:, :, 1].flatten()
    all_z = trajectories[:, :, 2].flatten()
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # 轨迹尾巴长度（显示过去多少步的轨迹）
    tail_len = 500
    
    def update(frame):
        for i in range(3):
            # 绘制尾巴
            start_idx = max(0, frame - tail_len)
            
            # 更新轨迹线
            lines[i].set_data(trajectories[i, start_idx:frame+1, 0], trajectories[i, start_idx:frame+1, 1])
            lines[i].set_3d_properties(trajectories[i, start_idx:frame+1, 2])
            
            # 更新当前位置点
            points[i].set_data([trajectories[i, frame, 0]], [trajectories[i, frame, 1]])
            points[i].set_3d_properties([trajectories[i, frame, 2]])
        
        ax.set_title(f'三体系统动态模拟 (时间: {time_points[frame]:.2f})')
        return lines + points
        
    # 创建动画
    ani = FuncAnimation(fig, update, frames=n_points, interval=20, blit=False)
    plt.show()

def interactive_simulation():
    """交互式模拟函数"""
    print("三体系统模拟")
    print("=" * 50)
    
    # 用户可以选择不同的初始条件
    print("请选择初始配置:")
    print("1. 等边三角形配置（默认）")
    print("2. 八字形轨道（稳定周期解）")
    print("3. 随机初始条件")
    
    choice = input("请输入选择 (1-3, 默认1): ").strip()
    
    if choice == "2":
        # 八字形轨道（稳定周期解）
        masses = [1.0, 1.0, 1.0]
        initial_positions = [
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
        initial_velocities = [
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0]
        ]
        dt = 0.01
        total_time = 50
    elif choice == "3":
        # 随机初始条件
        np.random.seed(42)
        masses = [1.0, 1.0, 1.0]
        initial_positions = np.random.uniform(-1, 1, (3, 3))
        initial_velocities = np.random.uniform(-0.5, 0.5, (3, 3))
        dt = 0.005
        total_time = 30
    else:
        # 默认配置
        masses = [1.0, 1.0, 1.0]
        initial_positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0]
        ]
        initial_velocities = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [-0.5, 0.0, 0.0]
        ]
        dt = 0.01
        total_time = 100
    
    save_interval = int(input("输出间隔步数 (默认20): ") or "20")
    
    # 运行模拟
    system, trajectories, time_points = simulate_three_body(
        dt=dt, 
        total_time=total_time, 
        save_interval=save_interval
    )
    
    # 绘制结果
    # plot_trajectories(trajectories, time_points)
    
    # 动态可视化
    print("正在生成动画...")
    animate_trajectories(trajectories, time_points)
    
    # 输出最终状态
    print("\n模拟完成!")
    print("最终状态:")
    final_positions = system.get_positions()
    for i, pos in enumerate(final_positions):
        print(f"天体{i+1}: 位置({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# 运行模拟
if __name__ == "__main__":
    # 直接运行默认模拟
    print("运行三体系统模拟...")
    system, trajectories, time_points = simulate_three_body(
        dt=0.001,  # 毕达哥拉斯问题需要更小的时间步长，因为天体会非常靠近
        total_time=70, 
        save_interval=10
    )
    
    print("正在生成动画...")
    animate_trajectories(trajectories, time_points)
    
    # 或者运行交互式版本
    # interactive_simulation()