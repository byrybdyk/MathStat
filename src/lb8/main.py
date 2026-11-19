"""ЛР8: интервалы для вероятности и моделирование их покрытия."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


N = 1000
Q = 0.95
SEED = 63
SAMPLE_SIZES = np.array([10, 50, 100, 150, 200, 500])
PROBABILITIES = (0.1, 0.5, 0.9)
METHODS = {
    "exact": "Клоппер–Пирсон (бисекция)",
    "normal": "Нормальная аппроксимация (Вальд)",
    "wilson": "Уилсон",
    "agresti_coull": "Агрести–Коулл",
}


def clopper_pearson(m, n, confidence=Q):
    """Решение P_p(X >= m)=α/2 и P_p(X <= m)=α/2.

    Векторизованная бисекция на [0, 1], абсолютная погрешность < 1e-12.
    При m=0 нижняя граница 0; при m=n верхняя граница 1.
    """
    m = np.asarray(m, dtype=float)
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n должно быть положительным целым")
    if not np.all(np.isfinite(m) & (m >= 0) & (m <= n) & (m == np.floor(m))):
        raise ValueError("Нужно целое число успехов 0 <= m <= n")
    if not 0 < confidence < 1:
        raise ValueError("Нужно 0 < Q < 1")
    tail = (1 - confidence) / 2
    a, b = np.zeros_like(m), np.ones_like(m)
    for _ in range(42):
        mid = (a + b) / 2
        below = stats.binom.sf(m - 1, n, mid) < tail
        a, b = np.where(below, mid, a), np.where(below, b, mid)
    lower = np.where(m == 0, 0.0, (a + b) / 2)
    a, b = np.zeros_like(m), np.ones_like(m)
    for _ in range(42):
        mid = (a + b) / 2
        above = stats.binom.cdf(m, n, mid) > tail
        a, b = np.where(above, mid, a), np.where(above, b, mid)
    upper = np.where(m == n, 1.0, (a + b) / 2)
    return lower, upper


def interval(m, n, method):
    if method == "exact":
        return clopper_pearson(m, n)
    return proportion_confint(m, n, alpha=1 - Q, method=method)


def compare_methods():
    print("ЛАБОРАТОРНАЯ РАБОТА №8")
    print(f"Q={Q}, N={N}, seed={SEED}")
    print("Сравнение при n=100: максимальное отклонение границ от бисекции")
    m = np.arange(101)
    exact = np.array(clopper_pearson(m, 100))
    reference = np.array(proportion_confint(m, 100, alpha=1-Q, method="beta"))
    error = np.max(np.abs(exact - reference))
    print(f"statsmodels beta (тоже точный Клоппер–Пирсон): {error:.3e}")
    if error > 1e-10:
        raise AssertionError("Бисекция не совпала с эталоном")
    for method in METHODS:
        bounds = np.array(interval(m, 100, method))
        print(f"{METHODS[method]}: {np.max(np.abs(bounds-exact)):.6f}")
    print("\nПримеры точных границ, n=100:")
    for successes in (0, 1, 10, 50, 90, 99, 100):
        low, high = clopper_pearson(successes, 100)
        print(f"m={successes:3}: [{low:.6f}; {high:.6f}]")


def simulate():
    rng = np.random.default_rng(SEED)
    results = {p: {method: [] for method in METHODS} for p in PROBABILITIES}
    for n in SAMPLE_SIZES:
        # Границы зависят только от m: считаем таблицу один раз для каждого n.
        tables = {method: interval(np.arange(n + 1), int(n), method) for method in METHODS}
        for p in PROBABILITIES:
            counts = rng.binomial(n, p, size=N)
            for method, (low, high) in tables.items():
                covered = (low <= p) & (p <= high)
                hits = int(np.count_nonzero(covered[counts]))
                ci = clopper_pearson(hits, N)
                theoretical = float(np.sum(stats.binom.pmf(np.arange(n+1), n, p)[covered]))
                results[p][method].append({"hits": hits, "coverage": hits/N,
                                           "ci": ci, "theoretical": theoretical})
    return results


def print_results(results):
    for p in PROBABILITIES:
        for method, rows in results[p].items():
            print(f"\np={p}: {METHODS[method]}")
            print("   n  Попаданий     Q*       95% ДИ для Q*     Q внутри?  Точное покрытие")
            for n, row in zip(SAMPLE_SIZES, rows):
                low, high = row["ci"]
                inside = "да" if low <= Q <= high else "нет"
                print(f"{n:4}  {row['hits']:4}/{N}  {row['coverage']:.4f}  "
                      f"[{low:.4f}; {high:.4f}]  {inside:>9}  {row['theoretical']:.6f}")
    print("\nДИ для Q* — отдельные 95% интервалы, не совместная полоса.")
    print("Попадание Q в ДИ означает совместимость с моделированием, а не доказательство равенства.")
    print("Точный Клоппер–Пирсон консервативен из-за дискретности; покрытие может превышать Q.")
    print("Переключение страниц: ← / → или кнопки Назад / Вперёд.")


class LabViewer:
    def __init__(self, results):
        self.results = results
        self.pages = ["width", "bounds", *METHODS]
        self.current_page = 0
        self.fig = plt.figure(figsize=(17, 7))
        self.previous_button = Button(self.fig.add_axes((0.30, 0.04, 0.16, 0.07)), "← Назад")
        self.next_button = Button(self.fig.add_axes((0.54, 0.04, 0.16, 0.07)), "Вперёд →")
        self.previous_button.on_clicked(self.previous_page)
        self.next_button.on_clicked(self.next_page)
        for action in ("back", "forward"):
            key = f"keymap.{action}"
            plt.rcParams[key] = [k for k in plt.rcParams[key] if k not in ("left", "right")]
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.axes = []
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
        for ax in self.axes:
            ax.remove()
        page = self.pages[self.current_page]
        grid = self.fig.add_gridspec(1, 3 if page in METHODS else 1,
                                    left=0.065, right=0.98, bottom=0.23, top=0.78, wspace=0.30)
        self.axes = [self.fig.add_subplot(cell) for cell in grid]
        if page in ("width", "bounds"):
            ax = self.axes[0]
            m = np.arange(101)
            for method, label in METHODS.items():
                low, high = interval(m, 100, method)
                if page == "width":
                    ax.plot(m, high-low, label=label)
                else:
                    line, = ax.plot(m, low, label=label)
                    ax.plot(m, high, "--", color=line.get_color())
            ax.set_xlabel("Количество успехов m, n = 100")
            ax.set_ylabel("Ширина интервала" if page == "width" else "Границы интервала (нижняя — сплошная, верхняя — пунктир)")
            title = "Ширина доверительного интервала от числа успехов" if page == "width" else "Сравнение границ доверительных интервалов"
        else:
            title = f"Покрытие Q* от n · {METHODS[page]}"
            for ax, p in zip(self.axes, PROBABILITIES):
                rows = self.results[p][page]
                values = np.array([r["coverage"] for r in rows])
                bounds = np.array([r["ci"] for r in rows])
                inside = (bounds[:, 0] <= Q) & (Q <= bounds[:, 1])
                ax.plot(SAMPLE_SIZES, values, color="gray", alpha=0.5)
                for mask, color, label in ((inside, "tab:blue", "Q внутри ДИ"),
                                            (~inside, "tab:red", "Q вне ДИ")):
                    ax.errorbar(SAMPLE_SIZES[mask], values[mask],
                                yerr=np.array([values[mask]-bounds[mask, 0], bounds[mask, 1]-values[mask]]),
                                fmt="o", capsize=5, color=color, label=label)
                ax.plot(SAMPLE_SIZES, [r["theoretical"] for r in rows], ":",
                        color="tab:green", label="Точное покрытие (сумма PMF)")
                ax.axhline(Q, color="black", linestyle="--", label=f"Q = {Q}")
                ax.set_title(f"p = {p}")
                ax.set_xlabel("Количество испытаний n")
                ax.set_ylabel("Доля покрытий Q* и её 95% ДИ")
                ax.set_ylim(max(0, min(bounds[:, 0].min(), Q)-0.04), 1.02)
        for ax in self.axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        self.fig.suptitle(f"Лабораторная работа №8 · Q={Q}, N={N}\n{title}\n"
                          f"Страница {self.current_page+1} из {len(self.pages)}", fontsize=13)
        self.fig.canvas.draw_idle()


def main():
    compare_methods()
    results = simulate()
    print_results(results)
    viewer = LabViewer(results)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
