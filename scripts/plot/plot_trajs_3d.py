import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_trajectories(file_paths, labels=None, title="Trajectory Plot"):
    """
    绘制多个3D轨迹文件

    Args:
        file_paths (list[str]): txt文件路径列表，每个文件是一个轨迹
        labels (list[str], optional): 每条轨迹对应的标签（图例中显示）
        title (str): 图表标题
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for idx, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"文件不存在：{path}")
            continue

        try:
            data = np.loadtxt(path)
            x = data[:, 2]
            y = data[:, 1]
            z = data[:, 3]
            label = labels[idx] if labels and idx < len(labels) else f'Traj {idx+1}'
            ax.plot(x, y, z, label=label)
        except Exception as e:
            print(f"读取或绘制 {path} 出错: {e}")

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.show()


# 示例用法
if __name__ == "__main__":
    files = [
        './output/vins_visual_ref2.txt',
        './output/vins_visual_ref1.txt',
        #  '/media/shentao/sunwenzSE/KITTYdatasets/2011_10_03_drive_0027/2011_10_03/2011_10_03_drive_0027_sync/result/truth_result.txt',
        # './data/navstate.txt',
    ]
    plot_trajectories(files, labels=["vins_visual_ref2", "vins_visual_ref1"])