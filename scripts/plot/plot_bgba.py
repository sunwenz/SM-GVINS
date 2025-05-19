import numpy as np
import matplotlib.pyplot as plt
import os

def plot_biases_over_time(file_path, title_prefix="Bias Over Time"):
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return

    try:
        data = np.loadtxt(file_path)
        timestamps = data[:, 0]
        bg = data[:, -6:-3]  # 倒数第6、5、4列
        ba = data[:, -3:]    # 倒数第3、2、1列

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(timestamps, bg[:, 0], label='bg_x')
        axes[0].plot(timestamps, bg[:, 1], label='bg_y')
        axes[0].plot(timestamps, bg[:, 2], label='bg_z')
        axes[0].set_ylabel('Gyro Bias [rad/s]')
        axes[0].set_title(f"{title_prefix}: Gyroscope Bias (bg)")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(timestamps, ba[:, 0], label='ba_x')
        axes[1].plot(timestamps, ba[:, 1], label='ba_y')
        axes[1].plot(timestamps, ba[:, 2], label='ba_z')
        axes[1].set_ylabel('Accel Bias [m/s²]')
        axes[1].set_title(f"{title_prefix}: Accelerometer Bias (ba)")
        axes[1].set_xlabel('Timestamp')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"读取或绘图出错: {e}")


# 示例调用
if __name__ == "__main__":
    plot_biases_over_time('./result/gins_preintg.txt')
