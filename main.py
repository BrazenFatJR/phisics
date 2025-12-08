import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Исходные данные
# --------------------------

r = 0.045       # внутренний радиус, м
R = 0.10        # внешний радиус, м
L = 0.18        # длина конденсатора, м
Vx = 5.5e6      # начальная скорость вдоль оси x, м/с

e = 1.6e-19     # заряд электрона, Кл
m = 9.11e-31    # масса электрона, кг

# --------------------------
# 1. Начальные условия
# --------------------------

y0 = (R - r) / 2
print("Начальное отклонение y0 =", y0, "м")

# --------------------------
# 2. Время пролёта
# --------------------------

t = L / Vx
print("Время пролёта t =", t, "с")

# --------------------------
# 3. Требуемое ускорение
# --------------------------

dy = r - y0
a = 2 * dy / t**2
print("Ускорение a =", a, "м/с^2")

# --------------------------
# 4. Минимальное напряжение
# --------------------------

rho0 = r + y0
U = a * m * rho0 * np.log(R / r) / e

print("Минимальная разность потенциалов U =", U, "В")

# --------------------------
# 5. Конечная скорость
# --------------------------

Vy_final = a * t
V_final = np.sqrt(Vx**2 + Vy_final**2)

print("Поперечная скорость Vy =", Vy_final, "м/с")
print("Конечная скорость Vкон =", V_final, "м/с")

# --------------------------
# 6. Построение графиков
# --------------------------

# График y(x)
x_vals = np.linspace(0, L, 400)
y_vals = y0 + 0.5 * a * (x_vals / Vx)**2

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals)
plt.xlabel("x, м")
plt.ylabel("y(x), м")
plt.title("Траектория y(x)")
plt.grid()
plt.show()

# График Vy(t)
t_vals = np.linspace(0, t, 400)
Vy_vals = a * t_vals

plt.figure(figsize=(6,4))
plt.plot(t_vals, Vy_vals)
plt.xlabel("t, c")
plt.ylabel("Vy(t), м/с")
plt.title("Скорость Vy(t)")
plt.grid()
plt.show()

# График ay(t)
ay_vals = a * np.ones_like(t_vals)

plt.figure(figsize=(6,4))
plt.plot(t_vals, ay_vals)
plt.xlabel("t, c")
plt.ylabel("ay(t), м/с^2")
plt.title("Ускорение ay(t)")
plt.grid()
plt.show()

# График y(t)
y_t_vals = y0 + 0.5 * a * t_vals**2

plt.figure(figsize=(6,4))
plt.plot(t_vals, y_t_vals)
plt.xlabel("t, c")
plt.ylabel("y(t), м")
plt.title("Координата y(t)")
plt.grid()
plt.show()
