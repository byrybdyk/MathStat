"""ЛР6. Доверительные интервалы Стьюдента и их фактическое покрытие."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import stats


N = 1000
Q = 0.95
MU = 0.0
SIGMA2 = 1.0  # Произвольная положительная дисперсия нормального распределения.
SEED = 85
SAMPLE_SIZES = np.array([10, 20, 30, 50, 60])
CONFIDENCE_LEVELS = np.array([0.80, 0.90, 0.95, 0.99])
DISTRIBUTIONS = ("Нормальное N(0, 1)", "Равномерное U[-1, 1]")


def student_interval(samples, confidence=Q):
    """Границы x̄ ± t((1 + Q)/2, n − 1) · s / √n вдоль последней оси.

    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 0 or samples.shape[-1] < 2:
        raise ValueError("В выборке должно быть не менее двух наблюдений")
    if not 0 < confidence < 1:
        raise ValueError("Доверительная вероятность должна лежать между 0 и 1")
    n = samples.shape[-1]
    mean = samples.mean(axis=-1)
    half_width = (
        stats.t.ppf((1 + confidence) / 2, df=n - 1)
        * samples.std(axis=-1, ddof=1) / np.sqrt(n)
    )
    return mean - half_width, mean + half_width


def simulate():
    rng = np.random.default_rng(SEED)
    results = {}
    for distribution in DISTRIBUTIONS:
        rows = []
        for n in SAMPLE_SIZES:
            if distribution == DISTRIBUTIONS[0]:
                samples = rng.normal(MU, np.sqrt(SIGMA2), size=(N, n))
            else:
                samples = rng.uniform(-1, 1, size=(N, n))
            lower, upper = student_interval(samples)
            hits = int(np.count_nonzero((lower <= MU) & (MU <= upper)))
            # Одни и те же выборки для всех Q: сравниваем влияние только Q.
            widths = np.array([
                np.mean(hi - lo)
                for lo, hi in (student_interval(samples, q)
                               for q in CONFIDENCE_LEVELS)
            ])
            rows.append({
                "n": int(n), "lower": lower, "upper": upper,
                "hits": hits, "coverage": hits / N, "widths": widths,
            })
        results[distribution] = rows
    return results


def print_results(results):
    print("\nЛАБОРАТОРНАЯ РАБОТА №6")
    print(f"N = {N}, Q = {Q}, seed = {SEED}, дисперсия нормального = {SIGMA2}")
    print("Интервал: x̄ ± t((1 + Q)/2; n − 1) · s / √n; s² с делителем n − 1")
    for name, rows in results.items():
        print(f"\n{name}: интервалы для первой выборки при каждом n")
        print(f"{'n':>5} {'Нижняя граница':>16} {'Верхняя граница':>16} {'Ширина':>12}")
        for row in rows:
            lo, hi = row["lower"][0], row["upper"][0]
            print(f"{row['n']:5d} {lo:16.6f} {hi:16.6f} {hi - lo:12.6f}")
        print("\nФактическое покрытие математического ожидания μ = 0")
        print(f"{'n':>5} {'Попаданий':>12} {'Всего':>8} {'Q факт.':>10} {'Q факт. − Q':>14}")
        for row in rows:
            print(f"{row['n']:5d} {row['hits']:12d} {N:8d} "
                  f"{row['coverage']:10.4f} {row['coverage'] - Q:14.4f}")
        print("\nСредняя ширина по 1000 выборкам")
        print(f"{'n':>5}" + "".join(f"{'Q=' + str(q):>12}" for q in CONFIDENCE_LEVELS))
        for row in rows:
            print(f"{row['n']:5d}" + "".join(f"{w:12.6f}" for w in row["widths"]))
    print("\nПри увеличении n ширина в среднем уменьшается, при увеличении Q растёт.")
    print("Для нормального распределения интервал имеет точное покрытие Q.")
    print("Для равномерного распределения покрытие приближённое.")
    print("1000 испытаний дают оценку покрытия, а не его точное значение.")
    print(f"При Q=0.95 стандартная ошибка оценки ≈ {np.sqrt(Q * (1 - Q) / N):.4f}.")
    print("Переключение графиков: ← / → или кнопки «Назад» / «Вперёд».")


class LabViewer:
    def __init__(self, results):
        self.results = results
        self.pages = [(kind, None) for kind in ("examples", "size", "confidence", "coverage")]
        # self.pages += [("intervals", int(n)) for n in SAMPLE_SIZES]
        self.current_page = 0
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.subplots_adjust(bottom=0.23, top=0.80, wspace=0.28)
        self.previous_button = Button(self.fig.add_axes((0.30, 0.05, 0.16, 0.07)), "← Назад")
        self.next_button = Button(self.fig.add_axes((0.54, 0.05, 0.16, 0.07)), "Вперёд →")
        self.previous_button.on_clicked(self.previous_page)
        self.next_button.on_clicked(self.next_page)
        # Убираем конфликт стрелок с навигацией панели инструментов Matplotlib.
        for action in ("back", "forward"):
            key = f"keymap.{action}"
            plt.rcParams[key] = [k for k in plt.rcParams[key] if k not in ("left", "right")]
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_page()

    def previous_page(self, event=None):
        self.current_page = (self.current_page - 1) % len(self.pages)
        self.update_page()

    def next_page(self, event=None):
        self.current_page = (self.current_page + 1) % len(self.pages)
        self.update_page()

    def on_key(self, event):
        if event.key == "left":
            self.previous_page()
        elif event.key == "right":
            self.next_page()

    def update_page(self):
        kind, n = self.pages[self.current_page]
        titles = {
            "examples": "Интервалы для отдельных выборок, Q = 0.95",
            "size": "Средняя ширина интервала в зависимости от n",
            "confidence": "Средняя ширина интервала в зависимости от Q",
            "coverage": "Фактическая доверительная вероятность, N = 1000, Q = 0.95",
            "intervals": f"Первые 100 из 1000 интервалов, n = {n}, Q = 0.95",
        }
        for ax, (name, rows) in zip(self.axes, self.results.items()):
            ax.clear()
            ax.set_title(name if name != DISTRIBUTIONS[0] else f"Нормальное N(0, {SIGMA2:g})")
            if kind == "examples":
                lo = np.array([r["lower"][0] for r in rows])
                hi = np.array([r["upper"][0] for r in rows])
                ax.errorbar(SAMPLE_SIZES, (lo + hi) / 2, yerr=(hi - lo) / 2,
                            fmt="o", capsize=6, label="Интервал Стьюдента")
                ax.axhline(MU, color="black", linestyle="--", label="Истинное μ = 0")
                ax.set(xlabel="Размер выборки n", ylabel="Математическое ожидание")
            elif kind == "size":
                for j, q in enumerate(CONFIDENCE_LEVELS):
                    ax.plot(SAMPLE_SIZES, [r["widths"][j] for r in rows], "o-", label=f"Q = {q}")
                ax.set(xlabel="Размер выборки n", ylabel="Средняя ширина интервала")
            elif kind == "confidence":
                for row in rows:
                    ax.plot(CONFIDENCE_LEVELS, row["widths"], "o-", label=f"n = {row['n']}")
                ax.set(xlabel="Доверительная вероятность Q", ylabel="Средняя ширина интервала")
                ax.set_xticks(CONFIDENCE_LEVELS)
            elif kind == "coverage":
                coverage = np.array([r["coverage"] for r in rows])
                # Центральные 95% значений частоты при истинном покрытии Q.
                low, high = stats.binom.ppf([0.025, 0.975], N, Q) / N
                ax.axhspan(low, high, alpha=0.15, color="gray", label="95% диапазон частоты при p = Q")
                ax.axhline(Q, color="black", linestyle="--", label="Назначенное Q = 0.95")
                ax.plot(SAMPLE_SIZES, coverage, "o-", label="Фактическое покрытие")
                for row in rows:
                    ax.annotate(f"{row['hits']}/{N}", (row["n"], row["coverage"]),
                                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
                ax.set(xlabel="Размер выборки n", ylabel="Доля интервалов, содержащих μ",
                       ylim=(min(low, coverage.min()) - 0.015, max(high, coverage.max()) + 0.025))
            # else:
            #     row = next(r for r in rows if r["n"] == n)
            #     lo, hi = row["lower"][:100], row["upper"][:100]
            #     hits = (lo <= MU) & (MU <= hi)
            #     indices = np.arange(1, len(lo) + 1)
            #     for mask, color, label in ((hits, "tab:blue", "Содержит μ"),
            #                                (~hits, "tab:red", "Не содержит μ")):
            #         ax.vlines(indices[mask], lo[mask], hi[mask], color=color, alpha=0.75, label=label)
            #     ax.axhline(MU, color="black", linestyle="--", label="Истинное μ = 0")
            #     ax.set(xlabel="Номер выборки", ylabel="Границы доверительного интервала")
            #     ax.text(0.02, 0.97, f"Всего попаданий: {row['hits']}/{N}",
            #             transform=ax.transAxes, va="top")
            if kind in ("examples", "size", "coverage"):
                ax.set_xticks(SAMPLE_SIZES)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        self.fig.suptitle("Лабораторная работа №6\n" + titles[kind]
                          + f"\nСтраница {self.current_page + 1} из {len(self.pages)}", fontsize=13)
        self.fig.canvas.draw_idle()


def main():
    results = simulate()
    print_results(results)
    viewer = LabViewer(results)  # Сохраняем ссылки на кнопки до закрытия окна.
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
