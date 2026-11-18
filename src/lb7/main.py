"""ЛР7: моделирование покрытия интервалов для среднего и медианы."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import stats


N = 1000
Q = 0.95
COVERAGE_CONFIDENCE = 0.95
SIGMA2 = 1.0
SEED = 47
SAMPLE_SIZES = np.array([10, 20, 30, 50, 60])
METHODS = {
    "asymptotic": "Асимптотический для среднего",
    "student": "Стьюдента для среднего",
    "hora": "Хора–Хора для среднего",
    "median": "Непараметрический для медианы",
}
DISTRIBUTIONS = ("normal", "uniform", "exponential")
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")


def distribution_name(distribution):
    return {
        "normal": f"Нормальное N(0, {SIGMA2:g})",
        "uniform": "Равномерное U[−1, 1]",
        "exponential": "Экспоненциальное Exp(1) − 1",
    }[distribution]


def true_value(distribution, method):
    if method == "median" and distribution == "exponential":
        return float(np.log(2) - 1)
    return 0.0


def validate_samples(samples, confidence):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 0 or samples.shape[-1] < 2:
        raise ValueError("Выборка должна содержать не менее двух наблюдений")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Наблюдения должны быть конечными числами")
    if not 0 < confidence < 1:
        raise ValueError("Доверительная вероятность должна лежать между 0 и 1")
    return samples


def mean_interval(samples, confidence=Q, student=False):
    """x̄ ± z·s/√n либо x̄ ± t·s/√n; s² имеет делитель n−1."""
    samples = validate_samples(samples, confidence)
    n = samples.shape[-1]
    quantile = (stats.t.ppf((1 + confidence) / 2, n - 1) if student
                else stats.norm.ppf((1 + confidence) / 2))
    half_width = quantile * samples.std(axis=-1, ddof=1) / np.sqrt(n)
    center = samples.mean(axis=-1)
    return center - half_width, center + half_width


def hora_interval(samples, confidence=Q, bounds=None):
    """Хора–Хора: x̄ ± (b−a)·d(n,Q), d — квантиль двусторонней KS-статистики.

    При известных конечных границах носителя [a,b] покрытие не меньше Q.
    bounds=None заменяет b−a выборочным размахом: это эвристика,
    не имеющая исходной гарантии покрытия. Нужна для исследования
    нормального и экспоненциального законов с неограниченным носителем.
    kstwo уже учитывает n: дополнительного деления на √n здесь нет.
    """
    samples = validate_samples(samples, confidence)
    if bounds is None:
        span = np.ptp(samples, axis=-1)
    else:
        a, b = bounds
        if not np.isfinite([a, b]).all() or a >= b:
            raise ValueError("Нужны конечные границы a < b")
        if np.any((samples < a) | (samples > b)):
            raise ValueError("Наблюдения выходят за заданные границы носителя")
        span = b - a
    half_width = span * stats.kstwo.ppf(confidence, samples.shape[-1])
    center = samples.mean(axis=-1)
    return center - half_width, center + half_width


def median_rank(n, confidence=Q):
    """Максимальное k с P(k ≤ Bin(n,1/2) ≤ n−k) ≥ Q, индексы с 1.

    Интервал [X_(k), X_(n−k+1)] — точный знаковый интервал.
    При k=0 берутся бесконечные границы. Из-за дискретности покрытие
    обычно больше Q, даже если исходное распределение непрерывно.
    """
    if n < 2 or not 0 < confidence < 1:
        raise ValueError("Требуются n ≥ 2 и 0 < Q < 1")
    candidates = np.arange(1, n // 2 + 1)
    valid = candidates[2 * stats.binom.cdf(candidates - 1, n, 0.5) <= 1 - confidence]
    return int(valid[-1]) if valid.size else 0


def median_interval(samples, confidence=Q):
    samples = validate_samples(samples, confidence)
    n = samples.shape[-1]
    k = median_rank(n, confidence)
    if k == 0:
        return (np.full(samples.shape[:-1], -np.inf),
                np.full(samples.shape[:-1], np.inf))
    ordered = np.sort(samples, axis=-1)
    return ordered[..., k - 1], ordered[..., n - k]


def coverage_interval(hits, total=N, confidence=COVERAGE_CONFIDENCE):
    """Точный биномиальный интервал Клоппера–Пирсона для вероятности покрытия."""
    interval = stats.binomtest(hits, total).proportion_ci(confidence, method="exact")
    return interval.low, interval.high


def simulate():
    if not np.isfinite(SIGMA2) or SIGMA2 <= 0:
        raise ValueError("Дисперсия нормального закона должна быть положительной")
    rng = np.random.default_rng(SEED)
    results = {}
    for distribution in DISTRIBUTIONS:
        rows = []
        for n in SAMPLE_SIZES:
            shape = (N, int(n))
            if distribution == "normal":
                samples = rng.normal(0, np.sqrt(SIGMA2), size=shape)
            elif distribution == "uniform":
                samples = rng.uniform(-1, 1, size=shape)
            else:
                samples = rng.exponential(1, size=shape) - 1
            # Все методы сравниваются на одних и тех же N выборках.
            intervals = {
                "asymptotic": mean_interval(samples),
                "student": mean_interval(samples, student=True),
                "hora": hora_interval(samples, bounds=(-1, 1) if distribution == "uniform" else None),
                "median": median_interval(samples),
            }
            methods = {}
            for method, (lower, upper) in intervals.items():
                target = true_value(distribution, method)
                hits = int(np.count_nonzero((lower <= target) & (target <= upper)))
                methods[method] = {
                    "lower": lower, "upper": upper, "target": target,
                    "hits": hits, "coverage": hits / N,
                    "coverage_ci": coverage_interval(hits),
                    "mean_width": float(np.mean(upper - lower)),
                }
            rows.append({"n": int(n), "methods": methods})
        results[distribution] = rows
    return results


def method_name(method, distribution):
    title = METHODS[method]
    if method == "hora" and distribution != "uniform":
        title += " (выборочный размах, эвристика)"
    return title


def print_results(results):
    print("\nЛАБОРАТОРНАЯ РАБОТА №7")
    print(f"N = {N}, Q = {Q}, доверие для оценки Q* = {COVERAGE_CONFIDENCE}, seed = {SEED}")
    print("Хора–Хора: для U[-1,1] известен носитель; для остальных — эвристика с размахом.")
    print("Для медианы Exp(1)−1 истинное значение ln(2)−1, для среднего — 0.")
    for distribution, rows in results.items():
        print(f"\n{distribution_name(distribution)}")
        for method in METHODS:
            print(f"\n{method_name(method, distribution)}")
            print(f"{'n':>4} {'Попаданий':>10} {'Q*':>8} {'95% ДИ для Q*':>22} "
                  f"{'Q внутри?':>10} {'Ср. ширина':>12} {'Первый интервал':>24}")
            for row in rows:
                data = row["methods"][method]
                low, high = data["coverage_ci"]
                inside = "да" if low <= Q <= high else "нет"
                print(f"{row['n']:4d} {data['hits']:5d}/{N:<4d} {data['coverage']:8.4f} "
                      f"[{low:.4f}; {high:.4f}] {inside:>14} {data['mean_width']:12.5f} "
                      f"[{data['lower'][0]:.5f}; {data['upper'][0]:.5f}]")
            if method == "median":
                print("Теоретическое покрытие знакового интервала (дискретность рангов):")
                for n in SAMPLE_SIZES:
                    k = median_rank(int(n))
                    exact = 1 - 2 * stats.binom.cdf(k - 1, n, 0.5)
                    print(f"n={n:2d}: k={k:2d}, покрытие={exact:.6f}")
    print("\nДИ для Q* — отдельные 95% интервалы Клоппера–Пирсона, не совместная полоса.")
    print("Попадание Q в ДИ означает совместимость с моделированием, а не доказательство равенства.")
    print("Ширину интервалов среднего и медианы следует сравнивать с учётом разных параметров.")
    print("Переключение страниц: ← / → или кнопки «Назад» / «Вперёд».")


class LabViewer:
    def __init__(self, results):
        self.results = results
        self.pages = [(kind, method) for kind in ("examples", "coverage") for method in METHODS]
        self.pages.append(("width", None))
        self.current_page = 0
        self.fig, self.axes = plt.subplots(1, 3, figsize=(17, 7))
        self.fig.subplots_adjust(left=0.055, right=0.98, bottom=0.25, top=0.76, wspace=0.30)
        self.note = self.fig.text(0.5, 0.165, "", ha="center", va="center", fontsize=9)
        self.previous_button = Button(self.fig.add_axes((0.30, 0.045, 0.16, 0.07)), "← Назад")
        self.next_button = Button(self.fig.add_axes((0.54, 0.045, 0.16, 0.07)), "Вперёд →")
        self.previous_button.on_clicked(self.previous_page)
        self.next_button.on_clicked(self.next_page)
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

    def draw_examples(self, ax, distribution, method, rows):
        # Первая выборка каждого n; те же самые выборки на страницах разных методов.
        data = [row["methods"][method] for row in rows]
        lower = np.array([d["lower"][0] for d in data])
        upper = np.array([d["upper"][0] for d in data])
        center = (lower + upper) / 2
        target = true_value(distribution, method)
        hits = (lower <= target) & (target <= upper)
        for mask, color, label in ((hits, "tab:blue", "Содержит параметр"),
                                    (~hits, "tab:red", "Не содержит параметр")):
            ax.errorbar(SAMPLE_SIZES[mask], center[mask],
                        yerr=(upper[mask] - lower[mask]) / 2, fmt="o",
                        capsize=6, color=color, label=label)
        symbol = "m" if method == "median" else "μ"
        ax.axhline(target, color="black", linestyle="--", label=f"Истинное {symbol} = {target:.5g}")
        ax.set_ylabel("Границы доверительного интервала")

    def draw_coverage(self, ax, distribution, method, rows):
        data = [row["methods"][method] for row in rows]
        coverage = np.array([d["coverage"] for d in data])
        bounds = np.array([d["coverage_ci"] for d in data])
        inside = (bounds[:, 0] <= Q) & (Q <= bounds[:, 1])
        ax.plot(SAMPLE_SIZES, coverage, color="gray", alpha=0.6)
        for mask, color, label in ((inside, "tab:blue", "Q попадает в ДИ для Q*"),
                                    (~inside, "tab:red", "Q вне ДИ для Q*")):
            ax.errorbar(SAMPLE_SIZES[mask], coverage[mask],
                        yerr=np.vstack((coverage[mask] - bounds[mask, 0],
                                        bounds[mask, 1] - coverage[mask])),
                        fmt="o", capsize=5, color=color, label=label)
        ax.axhline(Q, color="black", linestyle="--", label=f"Назначенное Q = {Q}")
        if method == "median":
            exact = [1 - 2 * stats.binom.cdf(median_rank(int(n)) - 1, n, 0.5)
                     for n in SAMPLE_SIZES]
            ax.plot(SAMPLE_SIZES, exact, ":", color="tab:green", label="Точное покрытие по рангам")
        for n, value in zip(SAMPLE_SIZES, coverage):
            ax.annotate(f"{value:.3f}", (n, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylim(max(0, min(bounds[:, 0].min(), Q) - 0.035), 1.025)
        ax.set_ylabel("Фактическое покрытие Q*")

    def update_page(self):
        kind, method = self.pages[self.current_page]
        for ax, distribution in zip(self.axes, DISTRIBUTIONS):
            ax.clear()
            rows = self.results[distribution]
            title = distribution_name(distribution)
            if method == "hora":
                title += "\n" + ("Известные границы [−1, 1]" if distribution == "uniform"
                                   else "Выборочный размах: эвристика")
            ax.set_title(title, fontsize=11)
            if kind == "examples":
                self.draw_examples(ax, distribution, method, rows)
            elif kind == "coverage":
                self.draw_coverage(ax, distribution, method, rows)
            else:
                for (key, label), color in zip(METHODS.items(), COLORS):
                    ax.plot(SAMPLE_SIZES, [r["methods"][key]["mean_width"] for r in rows],
                            "o-", color=color, label=label.replace(" для среднего", ""))
                ax.set_ylabel("Средняя ширина по 1000 выборкам")
            ax.set_xlabel("Размер выборки n")
            ax.set_xticks(SAMPLE_SIZES)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        title = {"examples": "Интервалы для отдельных выборок",
                 "coverage": "Зависимость Q* от n с 95% ДИ Клоппера–Пирсона",
                 "width": "Средняя ширина интервалов в зависимости от n"}[kind]
        subtitle = METHODS[method] if method else "Сравнение четырёх методов"
        self.fig.suptitle(f"Лабораторная работа №7 · Q = {Q}, N = {N}\n"
                          f"{title}\n{subtitle} · Страница {self.current_page + 1} из {len(self.pages)}",
                          fontsize=13)
        notes = []
        # if kind == "examples":
        #     notes.append("Для каждого n показан интервал по первой из 1000 выборок.")
        # elif kind == "coverage":
        # if method == "median" or kind == "width":
        #     # notes.append("Медиана Exp(1) − 1 равна ln(2) − 1; её интервал проверяется по медиане.")
        # if method == "hora" or kind == "width":
            # notes.append("Хора–Хора: для нормального и экспоненциального законов размах заменяет границы носителя; гарантия Q не действует.")
        self.note.set_text("\n".join(notes))
        self.fig.canvas.draw_idle()


def main():
    results = simulate()
    print_results(results)
    viewer = LabViewer(results)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
