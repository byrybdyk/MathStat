import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde, norm
from scipy.optimize import brentq

from concurrent.futures import ThreadPoolExecutor


# ============================================================
# ПАРАМЕТРЫ ЛАБОРАТОРНОЙ
# ============================================================

SAMPLE_SIZES = [10, 100, 1000, 10000]

# По условию N = 10^3.
# Для быстрой проверки можно временно поставить 50.
N_EXPERIMENTS = 1000

QUANTILES = [0.01, 0.05, 0.50]

RANDOM_SEED = 42


# ============================================================
# ГЕНЕРАЦИЯ ВЫБОРОК
# ============================================================

def generate_normal(n):
    """N(0, 1)."""
    return np.random.normal(0.0, 1.0, n)


def generate_uniform(n):
    """U[0, 1]."""
    return np.random.uniform(0.0, 1.0, n)


# ============================================================
# ЧИСЛО ПОЛОС ГИСТОГРАММЫ
# ============================================================

def calculate_bins(n):
    """
    k ≈ 1 + 1.59 * ln(n)
    """
    return max(2, int(round(1 + 1.59 * np.log(n))))


# ============================================================
# КВАНТИЛЬ ПО ЭМПИРИЧЕСКОЙ ФУНКЦИИ РАСПРЕДЕЛЕНИЯ
# ============================================================

def quantile_ecdf(sample, q):
    """
    Оценка квантили по выборочной функции распределения.
    """
    return np.quantile(sample, q)


# ============================================================
# ГИСТОГРАММА
# ============================================================

def create_histogram(sample):
    """
    Возвращает плотности, границы интервалов и центры интервалов.
    """
    k = calculate_bins(len(sample))

    densities, edges = np.histogram(
        sample,
        bins=k,
        density=True
    )

    centers = (edges[:-1] + edges[1:]) / 2

    return densities, edges, centers


# ============================================================
# КВАНТИЛЬ ПО ГИСТОГРАММЕ
# ============================================================

def quantile_histogram(sample, q):
    """
    Оценка квантили только по информации гистограммы.

    Внутри каждого интервала предполагается равномерное
    распределение.
    """
    densities, edges, _ = create_histogram(sample)

    widths = np.diff(edges)

    # Вероятности попадания в интервалы
    probabilities = densities * widths

    cumulative = np.cumsum(probabilities)

    index = np.searchsorted(cumulative, q)

    if index >= len(densities):
        index = len(densities) - 1

    previous_probability = (
        cumulative[index - 1]
        if index > 0
        else 0.0
    )

    current_probability = probabilities[index]

    if current_probability <= 0:
        return (edges[index] + edges[index + 1]) / 2

    fraction = (
        (q - previous_probability)
        / current_probability
    )

    return edges[index] + fraction * widths[index]


# ============================================================
# БЫСТРЫЙ КВАНТИЛЬ ПО KDE
# ============================================================

def kde_quantiles(sample, quantiles):
    """
    Быстрая оценка нескольких квантилей по KDE.

    Для гауссовского KDE функция распределения имеет вид:

        F_KDE(x) = 1/n * sum Phi((x - Xi) / h)

    поэтому нет необходимости строить сетку из 10 000 точек
    и численно интегрировать KDE.
    """

    kde = gaussian_kde(sample)

    # Ширина окна
    h = kde.factor * np.std(sample, ddof=1)

    # Защита для вырожденной выборки
    if h <= 1e-12:
        value = np.mean(sample)
        return np.array([value] * len(quantiles))

    def kde_cdf(x):
        z = (x - sample) / h
        return np.mean(norm.cdf(z))

    left = np.min(sample) - 5 * h
    right = np.max(sample) + 5 * h

    result = []

    for q in quantiles:
        value = brentq(
            lambda x: kde_cdf(x) - q,
            left,
            right
        )
        result.append(value)

    return np.array(result)


# ============================================================
# ОДИН ЭКСПЕРИМЕНТ
# ============================================================

def one_experiment(distribution, n):
    """
    Генерирует одну выборку и сразу оценивает все три
    квантили всеми тремя способами.
    """

    if distribution == "normal":
        sample = generate_normal(n)
    elif distribution == "uniform":
        sample = generate_uniform(n)
    else:
        raise ValueError("Неизвестное распределение")

    # ECDF
    ecdf_values = np.array([
        quantile_ecdf(sample, q)
        for q in QUANTILES
    ])

    # Гистограмма
    histogram_values = np.array([
        quantile_histogram(sample, q)
        for q in QUANTILES
    ])

    # KDE — создаём один KDE на выборку,
    # а не отдельный KDE для каждого квантиля.
    kde_values = kde_quantiles(
        sample,
        QUANTILES
    )

    return (
        ecdf_values,
        histogram_values,
        kde_values
    )


# ============================================================
# МНОГОКРАТНЫЙ ЭКСПЕРИМЕНТ
# ============================================================

def run_experiments(distribution, n, repetitions):
    """
    Повторяет эксперимент repetitions раз.

    Возвращает массивы размера:
        repetitions x 3

    Столбцы соответствуют:
        1%,
        5%,
        50%.
    """

    ecdf_values = np.empty(
        (repetitions, len(QUANTILES))
    )

    histogram_values = np.empty(
        (repetitions, len(QUANTILES))
    )

    kde_values = np.empty(
        (repetitions, len(QUANTILES))
    )

    for i in range(repetitions):

        (
            ecdf,
            histogram,
            kde
        ) = one_experiment(
            distribution,
            n
        )

        ecdf_values[i] = ecdf
        histogram_values[i] = histogram
        kde_values[i] = kde

    return (
        ecdf_values,
        histogram_values,
        kde_values
    )


# ============================================================
# МНОГОКРАТНЫЙ ЭКСПЕРИМЕНТ — ВСЕ РАСПРЕДЕЛЕНИЯ И n
# ============================================================

def run_all_experiments():
    """
    Полностью выполняет многократный эксперимент.

    Эта функция запускается в отдельном потоке, поэтому
    графики можно смотреть одновременно с вычислениями.
    """

    results = []

    total = (
        2
        * len(SAMPLE_SIZES)
        * N_EXPERIMENTS
    )

    experiment_counter = 0

    print("\n")
    print("=" * 70)
    print("МНОГОКРАТНЫЙ ЭКСПЕРИМЕНТ")
    print("=" * 70)

    for distribution in [
        "normal",
        "uniform"
    ]:

        for n in SAMPLE_SIZES:

            print(
                f"\nРаспределение: {distribution}, "
                f"n={n}"
            )

            (
                all_ecdf,
                all_hist,
                all_kde
            ) = run_experiments(
                distribution,
                n,
                N_EXPERIMENTS
            )

            experiment_counter += N_EXPERIMENTS

            print(
                f"  Выполнено "
                f"{experiment_counter}/{total}"
            )

            # Обрабатываем 1%, 5%, 50%
            for j, q in enumerate(QUANTILES):

                print(
                    f"  Квантиль "
                    f"{q * 100:.0f}%..."
                )

                ecdf_values = all_ecdf[:, j]
                hist_values = all_hist[:, j]
                kde_values = all_kde[:, j]

                q_true = true_quantile(
                    distribution,
                    q
                )

                # Средние оценки
                mean_ecdf = np.mean(
                    ecdf_values
                )

                mean_hist = np.mean(
                    hist_values
                )

                mean_kde = np.mean(
                    kde_values
                )

                # Дисперсии
                var_ecdf = np.var(
                    ecdf_values,
                    ddof=1
                )

                var_hist = np.var(
                    hist_values,
                    ddof=1
                )

                var_kde = np.var(
                    kde_values,
                    ddof=1
                )

                # Смещения
                bias_ecdf = (
                    mean_ecdf - q_true
                )

                bias_hist = (
                    mean_hist - q_true
                )

                bias_kde = (
                    mean_kde - q_true
                )

                results.append({

                    "Распределение":
                        distribution,

                    "n":
                        n,

                    "Квантиль":
                        q,

                    "Истинное значение":
                        q_true,

                    "Среднее ECDF":
                        mean_ecdf,

                    "Дисперсия ECDF":
                        var_ecdf,

                    "Смещение ECDF":
                        bias_ecdf,

                    "Среднее гистограмма":
                        mean_hist,

                    "Дисперсия гистограмма":
                        var_hist,

                    "Смещение гистограмма":
                        bias_hist,

                    "Среднее KDE":
                        mean_kde,

                    "Дисперсия KDE":
                        var_kde,

                    "Смещение KDE":
                        bias_kde
                })

    return pd.DataFrame(results)


# ============================================================
# ИСТИННЫЕ КВАНТИЛИ
# ============================================================

def true_quantile(distribution, q):

    if distribution == "normal":
        return norm.ppf(q)

    if distribution == "uniform":
        return q

    raise ValueError("Неизвестное распределение")


# ============================================================
# ГРАФИКИ С ПЕРЕКЛЮЧЕНИЕМ СТРЕЛКАМИ
# ============================================================

class GraphViewer:
    """
    Интерактивный просмотрщик графиков.

    Управление:
        ← / →  предыдущий / следующий график
        Home   первый график
        End    последний график
        Esc    закрыть окно

    Графики НЕ нужно закрывать вручную.
    """

    def __init__(self, plots):
        self.plots = plots
        self.index = 0
        self.fig = None

    def show(self):
        self.fig = plt.figure(
            figsize=(10, 6)
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        self.draw()

        plt.show()

    def draw(self):
        plt.clf()

        plot_function = self.plots[self.index]

        plot_function()

        plt.suptitle(
            f"График {self.index + 1} "
            f"из {len(self.plots)}",
            fontsize=10
        )

        plt.tight_layout()

        self.fig.canvas.draw_idle()

    def on_key(self, event):

        if event.key in ["right", "down", "space"]:
            self.index = (
                self.index + 1
            ) % len(self.plots)

            self.draw()

        elif event.key in ["left", "up"]:
            self.index = (
                self.index - 1
            ) % len(self.plots)

            self.draw()

        elif event.key == "home":
            self.index = 0
            self.draw()

        elif event.key == "end":
            self.index = len(self.plots) - 1
            self.draw()

        elif event.key == "escape":
            plt.close(self.fig)


# ============================================================
# СОЗДАНИЕ ГРАФИКА KDE + ГИСТОГРАММА
# ============================================================

def make_distribution_plot(
        distribution,
        n
):
    """
    Возвращает функцию, которая строит один график.
    """

    def plot():

        if distribution == "normal":
            sample = generate_normal(n)
        else:
            sample = generate_uniform(n)

        k = calculate_bins(n)

        plt.hist(
            sample,
            bins=k,
            density=True,
            alpha=0.5,
            edgecolor="black",
            label=f"Гистограмма, k={k}"
        )

        # KDE
        kde = gaussian_kde(sample)

        x_min = np.min(sample)
        x_max = np.max(sample)

        margin = 0.1 * (x_max - x_min)

        if margin == 0:
            margin = 1

        x = np.linspace(
            x_min - margin,
            x_max + margin,
            1000
        )

        plt.plot(
            x,
            kde(x),
            linewidth=2,
            label="KDE, гауссовское ядро"
        )

        # Истинная плотность
        if distribution == "normal":

            true_density = (
                np.exp(-x**2 / 2)
                / np.sqrt(2 * np.pi)
            )

            plt.plot(
                x,
                true_density,
                "--",
                linewidth=2,
                label="Истинная плотность N(0,1)"
            )

            title = (
                f"Нормальное распределение, n={n}"
            )

        else:

            true_density = np.where(
                (x >= 0) & (x <= 1),
                1.0,
                0.0
            )

            plt.plot(
                x,
                true_density,
                "--",
                linewidth=2,
                label="Истинная плотность U[0,1]"
            )

            title = (
                f"Равномерное распределение, n={n}"
            )

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("Плотность")

        plt.grid(alpha=0.3)
        plt.legend()

    return plot


# ============================================================
# СОЗДАНИЕ ГРАФИКОВ ДИСПЕРСИЙ
# ============================================================

def make_variance_plot(
        df,
        distribution,
        q
):
    """
    Возвращает функцию для одного графика дисперсий.
    """

    def plot():

        subset = df[
            (df["Распределение"] == distribution)
            & (df["Квантиль"] == q)
        ]

        plt.plot(
            subset["n"],
            subset["Дисперсия ECDF"],
            marker="o",
            label="Эмпирическая ФР"
        )

        plt.plot(
            subset["n"],
            subset["Дисперсия гистограмма"],
            marker="o",
            label="Гистограмма"
        )

        plt.plot(
            subset["n"],
            subset["Дисперсия KDE"],
            marker="o",
            label="KDE"
        )

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Размер выборки n")
        plt.ylabel("Дисперсия оценки квантили")

        plt.title(
            f"Дисперсия оценки {q * 100:.0f}% квантиля — "
            f"{distribution}"
        )

        plt.grid(alpha=0.3)
        plt.legend()

    return plot


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

def main():

    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("ЯДЕРНЫЕ ОЦЕНКИ ПЛОТНОСТИ РАСПРЕДЕЛЕНИЯ")
    print("=" * 70)

    print("\nПараметры:")
    print(f"Размеры выборок: {SAMPLE_SIZES}")
    print(f"Количество повторений N: {N_EXPERIMENTS}")
    print(f"Квантили: {QUANTILES}")

    # --------------------------------------------------------
    # ГРАФИКИ РАСПРЕДЕЛЕНИЙ
    # --------------------------------------------------------

    print("\nГенерируются графики распределений...")

    distribution_plots = []

    for distribution in [
        "normal",
        "uniform"
    ]:
        for n in SAMPLE_SIZES:

            distribution_plots.append(
                make_distribution_plot(
                    distribution,
                    n
                )
            )

    # --------------------------------------------------------
    # ЗАПУСК МНОГОКРАТНОГО ЭКСПЕРИМЕНТА В ФОНЕ
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ЗАПУСК МНОГОКРАТНОГО ЭКСПЕРИМЕНТА В ФОНЕ")
    print("=" * 70)

    executor = ThreadPoolExecutor(
        max_workers=1
    )

    experiment_future = executor.submit(
        run_all_experiments
    )

    print(
        "\nМногократный эксперимент запущен "
        "в фоновом режиме."
    )

    print(
        "Пока выполняются вычисления, "
        "можно смотреть графики."
    )

    # --------------------------------------------------------
    # ПРОСМОТР ГРАФИКОВ
    # --------------------------------------------------------

    print(
        "\nГрафики можно переключать стрелками:"
    )
    print("  → / ↓  следующий")
    print("  ← / ↑  предыдущий")
    print("  Home    первый")
    print("  End     последний")
    print("  Esc     закрыть")

    viewer = GraphViewer(
        distribution_plots
    )

    viewer.show()

    # --------------------------------------------------------
    # ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНИЯ МНОГОКРАТНОГО ЭКСПЕРИМЕНТА")
    print("=" * 70)

    if experiment_future.done():

        print(
            "\nМногократный эксперимент "
            "уже завершён."
        )

    else:

        print(
            "\nМногократный эксперимент "
            "ещё выполняется."
        )

        print(
            "Ожидание завершения вычислений..."
        )

    # Получение результата.
    # Если расчёты ещё не закончились,
    # здесь программа подождёт их завершения.
    df = experiment_future.result()

    executor.shutdown()

    print(
        "\nМногократный эксперимент завершён."
    )

    # --------------------------------------------------------
    # РЕЗУЛЬТАТЫ
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # СОХРАНЕНИЕ
    # --------------------------------------------------------

    df.to_csv(
        "results_lab2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    variance_table = df[
        [
            "Распределение",
            "n",
            "Квантиль",
            "Дисперсия ECDF",
            "Дисперсия гистограмма",
            "Дисперсия KDE"
        ]
    ]

    variance_table.to_csv(
        "variance_lab2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nРезультаты сохранены:"
    )
    print("  results_lab2.csv")
    print("  variance_lab2.csv")

    # --------------------------------------------------------
    # ГРАФИКИ ДИСПЕРСИЙ
    # --------------------------------------------------------

    variance_plots = []

    for distribution in [
        "normal",
        "uniform"
    ]:

        for q in QUANTILES:

            variance_plots.append(
                make_variance_plot(
                    df,
                    distribution,
                    q
                )
            )

    print(
        "\nТеперь откроется просмотрщик "
        "графиков дисперсий."
    )

    variance_viewer = GraphViewer(
        variance_plots
    )

    variance_viewer.show()

    print("\nРабота завершена.")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()
