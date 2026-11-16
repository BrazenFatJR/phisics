
# --------------------------
# 1. Библиотеки
# --------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Сохраняем графики в файлы, не открывая окна matplotlib.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


# --------------------------
# 2. Физические константы
# --------------------------

NM = 1e-9
ELECTRON_MASS = constants.m_e
COLORS = ["#2362A0", "#DB7A23", "#27935C", "#C34F54", "#7F5EB9"]


# --------------------------
# 3. Формулы для потенциальной ямы
# --------------------------

# Коэффициент перед второй производной в уравнении Шрёдингера (эВ·нм²).
def kinetic_coefficient(mass_factor: float = 1.0) -> float:
    if mass_factor <= 0:
        raise ValueError("Масса частицы должна быть положительной")
    return constants.hbar**2 / (2.0 * mass_factor * ELECTRON_MASS * NM**2 * constants.e)


# Потенциальная энергия: гладкий профиль sin² или прямоугольный профиль.
def potential(x, width: float, height: float, shape: str = "smooth"):
    x_arr = np.asarray(x, dtype=float)
    inside = (x_arr >= 0.0) & (x_arr <= width)
    if shape == "smooth":
        values = height * np.sin(np.pi * x_arr / width) ** 2
    elif shape == "square":
        values = np.full_like(x_arr, height)
    else:
        raise ValueError("Варианты формы потенциала: smooth, square")
    return np.where(inside, values, 0.0)


# --------------------------
# 4. Решение уравнения Шрёдингера
# --------------------------

# Находим энергии связанных состояний и, при необходимости, волновые функции.
def well_states(
    width: float,
    depth: float,
    mass_factor: float = 1.0,
    n_grid: int = 3000,
    shape: str = "smooth",
    with_functions: bool = True,
) -> dict:
    if min(width, depth) <= 0 or n_grid < 100:
        raise ValueError("Ширина и глубина должны быть > 0, число узлов >= 100")

    # Создаём равномерную сетку с запасом пространства за пределами ямы.
    margin = max(8.0, 2.5 * width)
    step = (width + 2.0 * margin) / (n_grid + 1)
    x = -margin + step * np.arange(1, n_grid + 1)
    alpha = kinetic_coefficient(mass_factor)

    # Для прямоугольной ямы учитываем долю ячейки, попавшую внутрь ямы.
    if shape == "square":
        cell_left = np.maximum(x - step / 2.0, 0.0)
        cell_right = np.minimum(x + step / 2.0, width)
        u = -depth * np.maximum(cell_right - cell_left, 0.0) / step
    else:
        u = -potential(x, width, depth, shape)

    # Собираем трёхдиагональную матрицу гамильтониана и ищем энергии ниже нуля.
    diag = 2.0 * alpha / step**2 + u
    offdiag = np.full(n_grid - 1, -alpha / step**2)
    result = eigh_tridiagonal(
        diag,
        offdiag,
        eigvals_only=not with_functions,
        select="v",
        select_range=(-depth - 1e-5, -1e-9),
        check_finite=False,
    )

    # Нормируем волновые функции так, чтобы интеграл |ψ|² по x равнялся 1.
    if with_functions:
        energies, eigvecs = result
        wavefunctions = np.empty_like(eigvecs)
        for i in range(eigvecs.shape[1]):
            wave = eigvecs[:, i]
            wave /= np.sqrt(trapezoid(np.abs(wave) ** 2, x))
            if wave[np.argmax(np.abs(wave))] < 0:
                wave = -wave
            wavefunctions[:, i] = wave
    else:
        energies = result
        wavefunctions = None

    return {
        "x": x,
        "potential": u,
        "energies": np.asarray(energies),
        "wavefunctions": wavefunctions,
        "step": step,
        "margin": margin,
    }


# --------------------------
# 5. Аналитическая проверка для прямоугольной ямы
# --------------------------

# Ищем разрешённые энергии по уравнениям для конечной прямоугольной ямы.
def square_well_energies_exact(
    width: float,
    depth: float,
    mass_factor: float = 1.0,
) -> np.ndarray:
    alpha = kinetic_coefficient(mass_factor)
    z0 = 0.5 * width * np.sqrt(depth / alpha)
    result = []

    for n in range(int(np.ceil(2.0 * z0 / np.pi)) + 1):
        left = n * np.pi / 2.0 + 1e-10
        right = min(z0, (n + 1) * np.pi / 2.0) - 1e-10
        if right <= left:
            break

        def root_equation(w):
            kappa_part = np.sqrt(max(0.0, z0**2 - w**2))
            return w - n * np.pi / 2.0 - np.arctan2(kappa_part, w)

        if root_equation(left) < 0.0 < root_equation(right):
            w = brentq(root_equation, left, right, xtol=1e-13)
            result.append(alpha * (2.0 * w / width) ** 2 - depth)

    return np.asarray(result)


# --------------------------
# 6. Настройка и сохранение графиков
# --------------------------

def setup_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.19,
        "savefig.dpi": 230,
    })


def save_plot(fig, output: Path):
    fig.savefig(output, facecolor="white")
    plt.close(fig)


# --------------------------
# 7. Построение графиков
# --------------------------

# График 1: форма потенциальной ямы и положения энергетических уровней.
def plot_levels(state: dict, width: float, depth: float, output: Path):
    x = state["x"]
    energies = state["energies"]
    visible = (x >= -width / 3.0) & (x <= 4.0 * width / 3.0)

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.plot(x[visible], state["potential"][visible],
            color="#23272D", lw=2.5, label="U(x)")
    for i, energy in enumerate(energies):
        ax.hlines(energy, 0.0, width, lw=1.9, color=COLORS[i % len(COLORS)],
                  label=f"E{i + 1} = {energy:.4f} эВ")
    ax.axhline(0, color="#666666", lw=0.9, ls=":")
    ax.set(
        xlim=(-width / 3.0, 4.0 * width / 3.0),
        ylim=(-1.10 * depth, 0.14 * depth),
        xlabel="Координата x, нм",
        ylabel="Энергия, эВ",
        title="Профиль ямы и дискретные уровни энергии",
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.96)
    fig.tight_layout()
    save_plot(fig, output)


# График 2: волновые функции для каждого связанного состояния.
def plot_wavefunctions(state: dict, width: float, output: Path):
    x = state["x"]
    energies = state["energies"]
    wavefunctions = state["wavefunctions"]
    visible = (x >= -width / 3.0) & (x <= 4.0 * width / 3.0)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axvspan(0.0, width, color="#D9E8F6", alpha=0.42)
    for i, energy in enumerate(energies):
        ax.plot(
            x[visible],
            wavefunctions[visible, i],
            lw=2.0,
            color=COLORS[i % len(COLORS)],
            label=f"ψ{i + 1}(x), E{i + 1} = {energy:.3f} эВ",
        )
    for boundary in (0.0, width):
        ax.axvline(boundary, color="#8A96A3", alpha=0.5, ls=":", lw=0.8)
    ax.axhline(0, color="#6A7887", lw=0.8)
    ax.set(
        xlim=(-width / 3.0, 4.0 * width / 3.0),
        xlabel="Координата x, нм",
        ylabel="Волновая функция ψ, нм⁻¹ᐟ²",
        title="Нормированные собственные волновые функции",
    )
    ax.legend(loc="upper right", fontsize=8.3)
    fig.tight_layout()
    save_plot(fig, output)


# График 3: плотность вероятности |ψ(x)|².
def plot_densities(state: dict, width: float, output: Path):
    x = state["x"]
    energies = state["energies"]
    wavefunctions = state["wavefunctions"]
    visible = (x >= -width / 3.0) & (x <= 4.0 * width / 3.0)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axvspan(0.0, width, color="#D9E8F6", alpha=0.42)
    for i, energy in enumerate(energies):
        ax.plot(
            x[visible],
            wavefunctions[visible, i] ** 2,
            lw=2.0,
            color=COLORS[i % len(COLORS)],
            label=f"|ψ{i + 1}|², E{i + 1} = {energy:.3f} эВ",
        )
    ax.set(
        xlim=(-width / 3.0, 4.0 * width / 3.0),
        xlabel="Координата x, нм",
        ylabel="Плотность вероятности, нм⁻¹",
        title="Распределение вероятности обнаружения электрона",
    )
    ax.legend(loc="upper right", fontsize=8.3)
    fig.tight_layout()
    save_plot(fig, output)


# График 4: как меняются уровни энергии при изменении ширины и глубины ямы.
def plot_parameter_points(
    width: float,
    depth: float,
    mass_factor: float,
    output: Path,
) -> dict:
    # Для каждой серии меняем только один параметр, другой оставляем неизменным.
    widths = np.linspace(0.6 * width, 1.4 * width, 9)
    depths = np.linspace(0.5 * depth, 1.5 * depth, 9)
    spectrum_by_width = [
        well_states(float(a), depth, mass_factor, 1800,
                    with_functions=False)["energies"]
        for a in widths
    ]
    spectrum_by_depth = [
        well_states(width, float(u), mass_factor, 1800,
                    with_functions=False)["energies"]
        for u in depths
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.25, 3.6), sharey=True)
    for ax, xs, spectrum in (
        (axes[0], widths, spectrum_by_width),
        (axes[1], depths, spectrum_by_depth),
    ):
        # Отмечаем рассчитанные уровни точками, без соединения линиями.
        for j in range(max(map(len, spectrum))):
            px = [xx for xx, row in zip(xs, spectrum) if j < len(row)]
            py = [float(row[j]) for row in spectrum if j < len(row)]
            ax.scatter(px, py, s=30, color=COLORS[j % len(COLORS)],
                       label=f"Уровень {j + 1}", zorder=3)
        ax.axhline(0, color="#707070", lw=0.8, ls=":")
        ax.grid(alpha=0.17)

    depth_label = f"{depth:.2f}".replace(".", ",")
    width_label = f"{width:.2f}".replace(".", ",")
    axes[0].set(
        xlabel="Ширина ямы a, нм",
        ylabel="Энергия E, эВ",
        title=f"Глубина U₀ = {depth_label} эВ",
    )
    axes[1].set(
        xlabel="Глубина ямы U₀, эВ",
        title=f"Ширина a = {width_label} нм",
    )
    axes[1].legend(loc="lower left", fontsize=7.8, ncol=2)
    fig.suptitle("Расчётные уровни при разных параметрах ямы", fontsize=12)
    fig.tight_layout()
    save_plot(fig, output)

    return {
        "widths_nm": widths.tolist(),
        "levels_by_width_ev": [row.tolist() for row in spectrum_by_width],
        "depths_ev": depths.tolist(),
        "levels_by_depth_ev": [row.tolist() for row in spectrum_by_depth],
    }


# --------------------------
# 8. Проверка точности вычислений
# --------------------------

# Сравниваем численное решение с аналитическим и смотрим сходимость сетки.
def check_square_well(width: float, depth: float, mass_factor: float) -> dict:
    exact = square_well_energies_exact(width, depth, mass_factor)
    convergence = []
    final_numeric = np.array([])

    for n_grid in (900, 1800, 3600, 7200):
        numeric = well_states(
            width, depth, mass_factor, n_grid, shape="square",
            with_functions=False,
        )
        count = min(len(exact), len(numeric["energies"]))
        errors = np.abs(exact[:count] - numeric["energies"][:count])
        convergence.append({
            "grid_points": n_grid,
            "step_nm": numeric["step"],
            "max_abs_error_ev": float(np.max(errors)) if count else None,
        })
        if n_grid == 7200:
            final_numeric = numeric["energies"]

    count = min(len(exact), len(final_numeric))
    return {
        "exact_energies_ev": exact.tolist(),
        "numeric_energies_ev": final_numeric.tolist(),
        "absolute_errors_ev": np.abs(exact[:count] - final_numeric[:count]).tolist(),
        "convergence": convergence,
    }


# Проверяем, что за пределами ямы волновые функции затухают экспоненциально.
def check_asymptotic_tail(
    state: dict,
    width: float,
    mass_factor: float,
) -> dict:
    x1 = width + 0.35
    x2 = width + 0.80
    x = state["x"]
    alpha = kinetic_coefficient(mass_factor)
    checks = []

    for i, energy in enumerate(state["energies"]):
        wave = state["wavefunctions"][:, i]
        psi1 = float(np.interp(x1, x, wave))
        psi2 = float(np.interp(x2, x, wave))
        numerical = abs(psi2 / psi1)
        kappa = float(np.sqrt(abs(energy) / alpha))
        analytical = float(np.exp(-kappa * (x2 - x1)))
        checks.append({
            "state": i + 1,
            "kappa_inverse_nm": kappa,
            "numerical_ratio": numerical,
            "analytical_ratio": analytical,
            "relative_error": abs(numerical / analytical - 1.0),
        })

    return {
        "points_nm": [x1, x2],
        "states": checks,
        "max_relative_error": max(item["relative_error"] for item in checks),
    }


# --------------------------
# 9. Основной запуск расчёта
# --------------------------

def run_simulation(args: argparse.Namespace) -> dict:
    if min(args.well_width, args.well_depth, args.mass) <= 0:
        raise ValueError("Ширина, глубина и масса должны быть положительными")
    if args.grid < 100:
        raise ValueError("Число узлов должно быть не меньше 100")

    # Подготавливаем папку для выходных файлов.
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_plots()

    # Вычисляем уровни энергии и волновые функции для заданной ямы.
    state = well_states(
        args.well_width, args.well_depth, args.mass, args.grid,
    )
    if not len(state["energies"]):
        raise RuntimeError("Не найдено связанных состояний для заданных параметров")

    # Строим четыре графика и выполняем две проверки точности.
    plot_levels(state, args.well_width, args.well_depth, output / "01_levels.png")
    plot_wavefunctions(state, args.well_width, output / "02_wavefunctions.png")
    plot_densities(state, args.well_width, output / "03_densities.png")
    sweep = plot_parameter_points(
        args.well_width, args.well_depth, args.mass,
        output / "04_parameter_points.png",
    )
    square_check = check_square_well(
        args.well_width, args.well_depth, args.mass,
    )
    tail_check = check_asymptotic_tail(
        state, args.well_width, args.mass,
    )

    # Собираем численные результаты и сохраняем их в JSON.
    result = {
        "parameters": {
            "well_width_nm": args.well_width,
            "well_depth_ev": args.well_depth,
            "mass_in_electron_masses": args.mass,
            "grid_points": args.grid,
            "kinetic_coefficient_ev_nm2": kinetic_coefficient(args.mass),
            "potential": "-U0*sin(pi*x/a)^2 inside [0,a], zero outside",
        },
        "smooth_well": {
            "energies_ev": state["energies"].tolist(),
            "grid_step_nm": state["step"],
            "margin_nm": state["margin"],
        },
        "parameter_sweep": sweep,
        "square_well_validation": square_check,
        "asymptotic_validation": tail_check,
        "graphs": [
            "01_levels.png",
            "02_wavefunctions.png",
            "03_densities.png",
            "04_parameter_points.png",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Выводим основные результаты в консоль.
    print("МОДУЛЬ 1 — ПОТЕНЦИАЛЬНАЯ ЯМА")
    print("Уровни энергии, эВ:",
          ", ".join(f"{e:.6f}" for e in state["energies"]))
    print("Максимальная ошибка для прямоугольной ямы, эВ:",
          f"{square_check['convergence'][-1]['max_abs_error_ev']:.3e}")
    print("Максимальная относительная ошибка асимптотики:",
          f"{tail_check['max_relative_error']:.4%}")
    print("Четыре графика и результаты сохранены в", output)
    return result


# --------------------------
# 10. Параметры запуска из командной строки
# --------------------------

# По умолчанию: ширина 3 нм, глубина 1 эВ, масса электрона, 3000 узлов.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Модуль 1. Связанные состояния частицы в потенциальной яме.",
    )
    parser.add_argument(
        "--well-width", type=float, default=3.0,
        help="Ширина потенциальной ямы в нм",
    )
    parser.add_argument(
        "--well-depth", type=float, default=1.0,
        help="Глубина потенциальной ямы в эВ",
    )
    parser.add_argument(
        "--mass", type=float, default=1.0,
        help="Масса в единицах массы электрона",
    )
    parser.add_argument(
        "--grid", type=int, default=3000,
        help="Количество внутренних узлов сетки",
    )
    parser.add_argument(
        "--output", default="results_well_module1",
        help="Каталог для четырёх графиков и results.json",
    )
    return parser


# --------------------------
# 11. Запуск программы
# --------------------------

if __name__ == "__main__":
    run_simulation(build_parser().parse_args())
