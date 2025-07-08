import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 실험에 사용된 파라미터 수
param_counts = [9415, 267527, 1059335]

# 각 지표별 CPU와 GPU 결과
total_time_cpu = [6.06918, 7.0305, 9.0288]
total_time_gpu = [6.97452, 7.22568, 7.35234]

learning_time_cpu = [1.371541, 2.198253, 3.97101]
learning_time_gpu = [1.933152, 2.176274, 2.261499]

cpu_exec_cpu = [11.30, 17.40, 26.20]
cpu_exec_gpu = [18.80, 19.70, 19.70]

kernel_time_cpu = [0.0, 0.0, 0.0]
kernel_time_gpu = [1.20, 1.60, 1.70]

# Total Time 그래프
plt.figure()
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
plt.plot(param_counts, total_time_cpu, label='CPU Total Time', marker='o')
plt.plot(param_counts, total_time_gpu, label='GPU Total Time', marker='o')
plt.xlabel('Parameter Count')
plt.ylabel('Total Time (s)')
plt.title('Total Time')
plt.legend()
plt.grid(True)
plt.show()

# Learning Time 그래프
plt.figure()
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
plt.plot(param_counts, learning_time_cpu, label='CPU Learning Time', marker='o')
plt.plot(param_counts, learning_time_gpu, label='GPU Learning Time', marker='o')
plt.xlabel('Parameter Count')
plt.ylabel('Learning Time (s)')
plt.title('Learning Time')
plt.legend()
plt.grid(True)
plt.show()

# CPU Execution Time 그래프
x = np.arange(len(param_counts))  # [0, 1, 2]
width = 0.35
plt.figure()
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
plt.bar(x - width/2, cpu_exec_cpu, width, label='CPU exec (CPU)')
plt.bar(x + width/2, cpu_exec_gpu, width, label='CPU exec (GPU)')
plt.xticks(x, param_counts)  # x축 라벨을 파라미터 수로
plt.xlabel('Parameter Count')
plt.ylabel('CPU Utilization (%)')
plt.title('CPU Execution Time')
plt.legend()
plt.grid(True)
plt.show()


# CUDA Kernel Time 그래프
plt.figure()
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
plt.bar(x - width/2, kernel_time_cpu, width, label='Kernel (CPU)')
plt.bar(x + width/2, kernel_time_gpu, width, label='Kernel (GPU)')
plt.xticks(x, param_counts)
plt.xlabel('Parameter Count')
plt.ylabel('CUDA Kernel Time (%)')
plt.title('CUDA Kernel Time')
plt.legend()
plt.grid(True)
plt.show()
