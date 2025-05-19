import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 state.txt 文件
data = np.loadtxt('/home/shentao/code/imu_map_constraint/data/navstate.txt')

# 提取 x、y 和 z 坐标
x = data[:, 2]
y = data[:, 1]
z = data[:, 3]

# 创建一个 3D 图形对象
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 轨迹
ax.plot(x, y, z, label='Trajectory')

# 设置标题和坐标轴标签
ax.set_title('Trajectory Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图例
ax.legend()

# 显示网格线
ax.grid(True)

# 显示图形
plt.show()