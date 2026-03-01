from RK4 import rk4
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义反应参数
kd = 10**(-5)   # 引发剂分解速率
kp = 145  # 增长速率
kt = 10**7   # 终止速率
f  = 0.5    # 引发效率

def kinetics_model(t,Y):
    I,M=Y[0],Y[1]
    I=max(0,I)
    M=max(0,M)
    dIdt=-kd*I
    dMdt=-kp*M*np.sqrt(f*kd*I/kt)

    return np.array([dIdt,dMdt])

y0 = [0.001, 10.0]  # 初始浓度 [I]=1.0, [M]=10.0
t0=0
t_end=50
h = 0.1           # 步长

t, results = rk4(kinetics_model, y0, t0 , t_end, h)

# 5. 可视化
plt.figure(figsize=(10, 5))
plt.plot(t, results[:, 0], label='[I] Initiator')
plt.plot(t, results[:, 1], label='[M] Monomer')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Polymerization Kinetics via Hand-coded RK4')
plt.legend()
plt.grid(False)
plt.show()