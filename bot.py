import numpy as np
import matplotlib.pyplot as plt


def wavelength_to_rgb(wavelength_nm):
    gamma = 0.8

    if 380 <= wavelength_nm < 440:
        r = -(wavelength_nm - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength_nm < 490:
        r = 0.0
        g = (wavelength_nm - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength_nm < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm < 580:
        r = (wavelength_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength_nm < 645:
        r = 1.0
        g = -(wavelength_nm - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wavelength_nm <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0

    if 380 <= wavelength_nm < 420:
        factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
    elif 420 <= wavelength_nm < 701:
        factor = 1.0
    elif 701 <= wavelength_nm <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength_nm) / (780 - 700)
    else:
        factor = 0.0

    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma

    return np.array([r, g, b])


def newton_intensity(r, R, wavelength):
    t = r ** 2 / (2 * R)
    return np.sin(2 * np.pi * t / wavelength) ** 2


R = float(input("Радиус кривизны линзы R, м: "))
lambda0_nm = float(input("Средняя длина волны lambda0, нм: "))
width_nm = float(input("Ширина спектра, нм: "))
r_max_mm = float(input("Максимальный радиус области, мм: "))

lambda0 = lambda0_nm * 1e-9
r_max = r_max_mm * 1e-3

N = 800

x = np.linspace(-r_max, r_max, N)
y = np.linspace(-r_max, r_max, N)
X, Y = np.meshgrid(x, y)
r_2d = np.sqrt(X ** 2 + Y ** 2)

mask = r_2d <= r_max

I_mono_2d = newton_intensity(r_2d, R, lambda0)
I_mono_2d[~mask] = 0

mono_color = wavelength_to_rgb(lambda0_nm)
image_mono = I_mono_2d[:, :, None] * mono_color[None, None, :]

if width_nm > 0:
    wavelengths_nm = np.linspace(
        lambda0_nm - width_nm / 2,
        lambda0_nm + width_nm / 2,
        41
    )
else:
    wavelengths_nm = np.array([lambda0_nm])

image_quasi = np.zeros((N, N, 3))
I_quasi_2d = np.zeros((N, N))

for lam_nm in wavelengths_nm:
    lam = lam_nm * 1e-9
    I = newton_intensity(r_2d, R, lam)
    color = wavelength_to_rgb(lam_nm)

    image_quasi += I[:, :, None] * color[None, None, :]
    I_quasi_2d += I

image_quasi /= len(wavelengths_nm)
I_quasi_2d /= len(wavelengths_nm)

image_quasi[~mask] = 0
I_quasi_2d[~mask] = 0

max_value = image_quasi.max()
if max_value > 0:
    image_quasi = image_quasi / max_value

max_value = image_mono.max()
if max_value > 0:
    image_mono = image_mono / max_value

r = np.linspace(0, r_max, 2000)
I_mono = newton_intensity(r, R, lambda0)

I_quasi = np.zeros_like(r)

for lam_nm in wavelengths_nm:
    lam = lam_nm * 1e-9
    I_quasi += newton_intensity(r, R, lam)

I_quasi /= len(wavelengths_nm)

m = np.arange(0, 10)
r_dark = np.sqrt(m * lambda0 * R)
r_bright = np.sqrt((m + 0.5) * lambda0 * R)

print()
print("Радиусы темных колец, мм:")
for number, value in zip(m, r_dark * 1000):
    print(f"m = {number}: {value:.4f}")

print()
print("Радиусы светлых колец, мм:")
for number, value in zip(m, r_bright * 1000):
    print(f"m = {number}: {value:.4f}")

plt.figure(figsize=(6, 6))
plt.imshow(
    image_mono,
    extent=[-r_max_mm, r_max_mm, -r_max_mm, r_max_mm],
    origin="lower"
)
plt.title(f"Монохроматический свет, λ = {lambda0_nm:.0f} нм")
plt.xlabel("x, мм")
plt.ylabel("y, мм")
plt.savefig("newton_mono.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(
    image_quasi,
    extent=[-r_max_mm, r_max_mm, -r_max_mm, r_max_mm],
    origin="lower"
)
plt.title(
    f"Квазимонохроматический свет, "
    f"{lambda0_nm - width_nm / 2:.0f}–{lambda0_nm + width_nm / 2:.0f} нм"
)
plt.xlabel("x, мм")
plt.ylabel("y, мм")
plt.savefig("newton_quasi.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(r * 1000, I_mono, label="монохроматический свет")
plt.plot(r * 1000, I_quasi, label="квазимонохроматический свет")
plt.xlabel("r, мм")
plt.ylabel("I / I0")
plt.title("Зависимость интенсивности от радиальной координаты")
plt.grid(True)
plt.legend()
plt.savefig("newton_intensity_graph.png", dpi=300)
plt.show()
