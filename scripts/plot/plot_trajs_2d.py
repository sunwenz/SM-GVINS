import numpy as np
import matplotlib.pyplot as plt
import os

def plot_trajectories_2d(file_paths, labels=None, title="2D Trajectory Plot"):
    """
    绘制多个2D轨迹文件（X–Y 平面）

    Args:
        file_paths (list[str]): txt文件路径列表，每个文件是一条轨迹
        labels (list[str], optional): 每条轨迹对应的标签（图例中显示）
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))

    for idx, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"文件不存在：{path}")
            continue

        try:
            data = np.loadtxt(path)
            # 假设文件列顺序和原来一致：time, y, x, z...
            x = data[:, 2]
            y = data[:, 1]
            label = labels[idx] if labels and idx < len(labels) else f'Traj {idx+1}'
            plt.plot(x, y, label=label)
        except Exception as e:
            print(f"读取或绘制 {path} 出错: {e}")

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保持 X/Y 比例
    plt.show()


# 示例用法
if __name__ == "__main__":
    files = [
        './output/euroc_state.txt',
    ]
    plot_trajectories_2d(files, labels=["ground_truth"])
