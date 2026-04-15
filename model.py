import numpy as np
import matplotlib.pyplot as plt
 
mu0 = 4 * np.pi * 1e-7
 
def turns_count(wire_length, frame_diameter, coil_length, wire_diameter):
    d_mean = frame_diameter + wire_diameter
    return np.sqrt(wire_length**2 - coil_length**2) / (np.pi * d_mean)
 
def magnetic_field_center(current, turns, coil_length, radius):
    n0 = turns / coil_length
    return mu0 * n0 * current * coil_length / (2 * np.sqrt(radius**2 + (coil_length / 2)**2))
 
def inductance(turns, coil_length, radius):
    s = np.pi * radius**2
    return mu0 * turns**2 * s / coil_length
 
def valid_single_layer(turns, coil_length, wire_diameter):
    step = coil_length / turns
    return step >= wire_diameter
 
def solve(L_wire, d_wire, D_frame, I, points=2000):
    r = D_frame / 2
    l_min = d_wire * 1.01
    l_max = L_wire * 0.999
 
    lengths = np.linspace(l_min, l_max, points)
    good_l = []
    good_B = []
    good_N = []
    good_Lind = []
 
    for l in lengths:
        N = turns_count(L_wire, D_frame, l, d_wire)
        if N <= 0:
            continue
        if not valid_single_layer(N, l, d_wire):
            continue
 
        B = magnetic_field_center(I, N, l, r)
        Lind = inductance(N, l, r)
 
        good_l.append(l)
        good_B.append(B)
        good_N.append(N)
        good_Lind.append(Lind)
 
    good_l = np.array(good_l)
    good_B = np.array(good_B)
    good_N = np.array(good_N)
    good_Lind = np.array(good_Lind)
 
    idx = np.argmax(good_B)
    return good_l, good_B, good_N, good_Lind, idx
 
L_wire = float(input("Введите длину провода L (м): "))
d_wire = float(input("Введите диаметр провода d (м): "))
D_frame = float(input("Введите диаметр каркаса D (м): "))
I = float(input("Введите ток I (А): "))
 
l, B, N, Lind, idx = solve(L_wire, d_wire, D_frame, I)
 
print()
print(f"Оптимальная длина катушки l_opt = {l[idx]:.6f} м")
print(f"Максимальное поле в центре B_max = {B[idx]:.6e} Тл")
print(f"Число витков N_opt = {N[idx]:.3f}")
print(f"Индуктивность L_opt = {Lind[idx]:.6e} Гн")
 
plt.figure(figsize=(8, 5))
plt.plot(l, B, label="B(l)")
plt.scatter(l[idx], B[idx], label="Максимум")
plt.xlabel("l, м")
plt.ylabel("B(l), Тл")
plt.title("Зависимость магнитного поля в центре катушки от ее длины")
plt.grid(True)
plt.legend()
plt.show()
