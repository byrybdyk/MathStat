import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from math import factorial


# ============================================================
# НАСТРОЙКИ ЛАБОРАТОРНОЙ
# ============================================================

RANDOM_SEED = 42

# Истинные параметры равномерного распределения U(a, b)
TRUE_A = 0.0
TRUE_B = 1.0

# Истинный параметр экспоненциального распределения
# f(x) = lambda * exp(-lambda*x)
TRUE_LAMBDA = 1.0

SAMPLE_SIZES = np.unique(
    np.linspace(10, 1000, 20).astype(int)
)

N_REPETITIONS = 1000

# Случайный генератор
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def sample_moments(x, max_order):
    """
    Выборочные начальные моменты:
    m_k = (1/n) * sum(x_i^k)
    """
    return np.array([
        np.mean(x ** k)
        for k in range(1, max_order + 1)
    ])


# ============================================================
# РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ
# ============================================================

def uniform_theoretical_moments(a, b, max_order):
    """
    Теоретические начальные моменты равномерного распределения
    на [a, b]:

        E[X^k] = (b^(k+1) - a^(k+1)) / ((k+1)(b-a))
    """
    moments = []

    for k in range(1, max_order + 1):
        value = (
            b ** (k + 1) - a ** (k + 1)
        ) / (
            (k + 1) * (b - a)
        )
        moments.append(value)

    return np.array(moments)


def uniform_moments_covariance(a, b, max_order):
    """
    Теоретическая ковариационная матрица
    для вектора (X, X^2, ..., X^k).
    """
    all_moments = uniform_theoretical_moments(
        a, b, 2 * max_order
    )

    covariance = np.zeros((max_order, max_order))

    for i in range(max_order):
        for j in range(max_order):
            # i+1 и j+1 — степени
            covariance[i, j] = (
                all_moments[i + j + 1]
                - all_moments[i] * all_moments[j]
            )

    return covariance


def uniform_mom_estimator(x):
    """
    Классический метод моментов.

    Используем первые два момента:

        E[X]   = (a+b)/2
        E[X^2] = (a^2+ab+b^2)/3

    Получаем:
        a = mean - sqrt(3 * (m2 - mean^2))
        b = mean + sqrt(3 * (m2 - mean^2))
    """
    m1 = np.mean(x)
    m2 = np.mean(x ** 2)

    variance = max(m2 - m1 ** 2, 0)

    half_width = np.sqrt(3 * variance)

    a_hat = m1 - half_width
    b_hat = m1 + half_width

    return a_hat, b_hat


def uniform_gmm_estimator(x, n_moments):
    moments = sample_moments(x, n_moments)

    # Весовая матрица рассчитывается один раз
    # для истинного распределения U(0, 1).
    covariance = uniform_moments_covariance(
        0.0, 1.0, n_moments
    )
    covariance += 1e-9 * np.eye(n_moments)

    L = np.linalg.cholesky(covariance)

    a0, b0 = uniform_mom_estimator(x)
    width0 = max(b0 - a0, 1e-3)

    # Оптимизируем a и log(b-a), чтобы b всегда было больше a.
    from scipy.optimize import least_squares

    def residual(theta):
        a = theta[0]
        width = np.exp(theta[1])
        b = a + width

        theoretical = uniform_theoretical_moments(
            a, b, n_moments
        )

        return np.linalg.solve(
            L, theoretical - moments
        )

    result = least_squares(
        residual,
        x0=np.array([a0, np.log(width0)]),
        bounds=(
            [-5.0, np.log(1e-4)],
            [5.0, np.log(10.0)]
        ),
        max_nfev=60,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7
    )

    a_hat = result.x[0]
    b_hat = a_hat + np.exp(result.x[1])

    return a_hat, b_hat

# ============================================================
# ЭКСПОНЕНЦИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ
# ============================================================

def exponential_theoretical_moments(lmbda, max_order):
    """
    Для экспоненциального распределения:

        E[X^k] = k! / lambda^k
    """

    moments = []

    for k in range(1, max_order + 1):
        moments.append(
            factorial(k) / lmbda ** k
        )

    return np.array(moments)


def exponential_moments_covariance(lmbda, max_order):
    """
    Ковариационная матрица моментов:
        Cov(X^i, X^j)
        = E[X^(i+j)] - E[X^i]E[X^j]
    """

    covariance = np.zeros((max_order, max_order))

    for i in range(1, max_order + 1):
        for j in range(1, max_order + 1):

            e_ij = (
                factorial(i + j)
                / lmbda ** (i + j)
            )

            e_i = (
                factorial(i)
                / lmbda ** i
            )

            e_j = (
               factorial(j)
                / lmbda ** j
            )

            covariance[i - 1, j - 1] = (
                e_ij - e_i * e_j
            )

    return covariance


def exponential_mom_estimator(x):
    """
    Классический метод моментов.

    E[X] = 1/lambda

    Поэтому:

        lambda_hat = 1 / mean(X)
    """

    return 1.0 / np.mean(x)


def exponential_gmm_estimator(x, n_moments):
    from scipy.optimize import least_squares
    from math import factorial

    # Нормируем моменты на k!, чтобы уменьшить
    # численные проблемы при использовании 7 моментов.
    moments = np.array([
        np.mean(x ** k) / factorial(k)
        for k in range(1, n_moments + 1)
    ])

    # Теоретические нормированные моменты:
    # E[X^k] / k! = 1 / lambda^k
    covariance = np.zeros((n_moments, n_moments))

    for i in range(1, n_moments + 1):
        for j in range(1, n_moments + 1):
            covariance[i - 1, j - 1] = (
                factorial(i + j)
                / (factorial(i) * factorial(j))
                - 1.0
            )

    # Регуляризация ковариационной матрицы
    covariance += 1e-9 * np.eye(n_moments)

    L = np.linalg.cholesky(covariance)

    lambda0 = 1.0 / np.mean(x)

    def residual(theta):
        lam = np.exp(theta[0])

        theoretical = np.array([
            1.0 / lam**k
            for k in range(1, n_moments + 1)
        ])

        return np.linalg.solve(
            L, theoretical - moments
        )

    result = least_squares(
        residual,
        x0=np.array([np.log(lambda0)]),
        bounds=(
            [np.log(1e-5)],
            [np.log(1e5)]
        ),
        max_nfev=60,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7
    )

    return np.exp(result.x[0])
# ============================================================
# ПУНКТ 1. РАВНОМЕРНОЕ: КЛАССИЧЕСКИЙ МЕТОД МОМЕНТОВ
# ============================================================

def calculate_uniform_classic():
    a_values = []
    b_values = []

    for n in SAMPLE_SIZES:

        x = rng.uniform(
            TRUE_A,
            TRUE_B,
            size=n
        )

        a_hat, b_hat = uniform_mom_estimator(x)

        a_values.append(a_hat)
        b_values.append(b_hat)

    return (
        np.array(a_values),
        np.array(b_values)
    )


# ============================================================
# ПУНКТ 2. РАВНОМЕРНОЕ: GMM
# ============================================================

def calculate_uniform_gmm():

    results = {}

    for k in [3, 5, 7]:

        a_values = []
        b_values = []

        print(f"Равномерное распределение: GMM, {k} моментов...")

        for n in SAMPLE_SIZES:

            x = rng.uniform(
                TRUE_A,
                TRUE_B,
                size=n
            )

            a_hat, b_hat = uniform_gmm_estimator(
                x,
                k
            )

            a_values.append(a_hat)
            b_values.append(b_hat)

        results[k] = {
            "a": np.array(a_values),
            "b": np.array(b_values)
        }

    return results


# ============================================================
# ПУНКТ 3. СКО ДЛЯ РАВНОМЕРНОГО
# ============================================================

def calculate_uniform_std():

    methods = {
        "МО": {
            "a": [],
            "b": []
        }
    }

    for k in [3, 5, 7]:
        methods[f"GMM {k}"] = {
            "a": [],
            "b": []
        }

    for n in SAMPLE_SIZES:

        print(
            f"СКО равномерного: n = {n}"
        )

        classic_a = []
        classic_b = []

        gmm_a = {
            3: [],
            5: [],
            7: []
        }

        gmm_b = {
            3: [],
            5: [],
            7: []
        }

        for _ in range(N_REPETITIONS):

            x = rng.uniform(
                TRUE_A,
                TRUE_B,
                size=n
            )

            # Классический МО
            a_hat, b_hat = uniform_mom_estimator(x)

            classic_a.append(a_hat)
            classic_b.append(b_hat)

            # GMM
            for k in [3, 5, 7]:

                a_hat, b_hat = uniform_gmm_estimator(
                    x,
                    k
                )

                gmm_a[k].append(a_hat)
                gmm_b[k].append(b_hat)

        methods["МО"]["a"].append(
            np.std(classic_a, ddof=1)
        )

        methods["МО"]["b"].append(
            np.std(classic_b, ddof=1)
        )

        for k in [3, 5, 7]:

            methods[f"GMM {k}"]["a"].append(
                np.std(gmm_a[k], ddof=1)
            )

            methods[f"GMM {k}"]["b"].append(
                np.std(gmm_b[k], ddof=1)
            )

    return methods


# ============================================================
# ПУНКТ 4. ЭКСПОНЕНЦИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ
# ============================================================

def calculate_exponential_classic():

    values = []

    for n in SAMPLE_SIZES:

        x = rng.exponential(
            scale=1 / TRUE_LAMBDA,
            size=n
        )

        values.append(
            exponential_mom_estimator(x)
        )

    return np.array(values)


def calculate_exponential_gmm():

    results = {}

    for k in [3, 5, 7]:

        values = []

        print(
            f"Экспоненциальное распределение: "
            f"GMM, {k} моментов..."
        )

        for n in SAMPLE_SIZES:

            x = rng.exponential(
                scale=1 / TRUE_LAMBDA,
                size=n
            )

            values.append(
                exponential_gmm_estimator(
                    x,
                    k
                )
            )

        results[k] = np.array(values)

    return results


def calculate_exponential_std():

    methods = {
        "МО": []
    }

    for k in [3, 5, 7]:
        methods[f"GMM {k}"] = []

    for n in SAMPLE_SIZES:

        print(
            f"СКО экспоненциального: n = {n}"
        )

        classic = []

        gmm = {
            3: [],
            5: [],
            7: []
        }

        for _ in range(N_REPETITIONS):

            x = rng.exponential(
                scale=1 / TRUE_LAMBDA,
                size=n
            )

            # Классический МО
            classic.append(
                exponential_mom_estimator(x)
            )

            # GMM
            for k in [3, 5, 7]:

                gmm[k].append(
                    exponential_gmm_estimator(
                        x,
                        k
                    )
                )

        methods["МО"].append(
            np.std(classic, ddof=1)
        )

        for k in [3, 5, 7]:

            methods[f"GMM {k}"].append(
                np.std(gmm[k], ddof=1)
            )

    return methods


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================

def create_figures(
    uniform_classic,
    uniform_gmm,
    uniform_std,
    exponential_classic,
    exponential_gmm,
    exponential_std
):

    figures = []

    # --------------------------------------------------------
    # 1. Равномерное — классический МО
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        SAMPLE_SIZES,
        uniform_classic[0],
        marker="o",
        markersize=3,
        label=r"$\hat a$"
    )

    ax.plot(
        SAMPLE_SIZES,
        uniform_classic[1],
        marker="o",
        markersize=3,
        label=r"$\hat b$"
    )

    ax.axhline(
        TRUE_A,
        linestyle="--",
        label="Истинное a = 0"
    )

    ax.axhline(
        TRUE_B,
        linestyle="--",
        label="Истинное b = 1"
    )

    ax.set_title(
        "Равномерное распределение U(0,1)\n"
        "Классический метод моментов"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel("Значение оценки")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "1 / 8 — Равномерное распределение: МО")
    )


    # --------------------------------------------------------
    # 2. Равномерное — GMM для a
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in [3, 5, 7]:

        ax.plot(
            SAMPLE_SIZES,
            uniform_gmm[k]["a"],
            marker="o",
            markersize=3,
            label=f"GMM, {k} моментов"
        )

    ax.axhline(
        TRUE_A,
        linestyle="--",
        label="Истинное значение a = 0"
    )

    ax.set_title(
        "Равномерное распределение\n"
        "GMM: оценка левой границы"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel(r"$\hat a$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "2 / 8 — Равномерное: GMM для a")
    )


    # --------------------------------------------------------
    # 3. Равномерное — GMM для b
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in [3, 5, 7]:

        ax.plot(
            SAMPLE_SIZES,
            uniform_gmm[k]["b"],
            marker="o",
            markersize=3,
            label=f"GMM, {k} моментов"
        )

    ax.axhline(
        TRUE_B,
        linestyle="--",
        label="Истинное значение b = 1"
    )

    ax.set_title(
        "Равномерное распределение\n"
        "GMM: оценка правой границы"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel(r"$\hat b$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "3 / 8 — Равномерное: GMM для b")
    )


    # --------------------------------------------------------
    # 4. Равномерное — СКО
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in uniform_std:

        ax.plot(
            SAMPLE_SIZES,
            uniform_std[method]["a"],
            marker="o",
            markersize=3,
            label=f"{method}: a"
        )

        ax.plot(
            SAMPLE_SIZES,
            uniform_std[method]["b"],
            marker="x",
            markersize=3,
            linestyle="--",
            label=f"{method}: b"
        )

    ax.set_title(
        "Равномерное распределение\n"
        "СКО оценок, 1000 выборок"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel("СКО")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "4 / 8 — Равномерное: СКО")
    )


    # --------------------------------------------------------
    # 5. Экспоненциальное — классический МО
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        SAMPLE_SIZES,
        exponential_classic,
        marker="o",
        markersize=3,
        label=r"$\hat\lambda$"
    )

    ax.axhline(
        TRUE_LAMBDA,
        linestyle="--",
        label=r"Истинное $\lambda = 1$"
    )

    ax.set_title(
        "Экспоненциальное распределение\n"
        "Классический метод моментов"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel(r"$\hat\lambda$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "5 / 8 — Экспоненциальное: МО")
    )


    # --------------------------------------------------------
    # 6. Экспоненциальное — GMM
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in [3, 5, 7]:

        ax.plot(
            SAMPLE_SIZES,
            exponential_gmm[k],
            marker="o",
            markersize=3,
            label=f"GMM, {k} моментов"
        )

    ax.axhline(
        TRUE_LAMBDA,
        linestyle="--",
        label=r"Истинное $\lambda = 1$"
    )

    ax.set_title(
        "Экспоненциальное распределение\n"
        "GMM: 3, 5 и 7 моментов"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel(r"$\hat\lambda$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "6 / 8 — Экспоненциальное: GMM")
    )


    # --------------------------------------------------------
    # 7. Экспоненциальное — СКО
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in exponential_std:

        ax.plot(
            SAMPLE_SIZES,
            exponential_std[method],
            marker="o",
            markersize=3,
            label=method
        )

    ax.set_title(
        "Экспоненциальное распределение\n"
        "СКО оценок, 1000 выборок"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel("СКО")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "7 / 8 — Экспоненциальное: СКО")
    )


    # --------------------------------------------------------
    # 8. Сравнение СКО
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        SAMPLE_SIZES,
        uniform_std["МО"]["a"],
        label="U(0,1): МО, a"
    )

    ax.plot(
        SAMPLE_SIZES,
        uniform_std["GMM 3"]["a"],
        label="U(0,1): GMM 3, a"
    )

    ax.plot(
        SAMPLE_SIZES,
        uniform_std["GMM 5"]["a"],
        label="U(0,1): GMM 5, a"
    )

    ax.plot(
        SAMPLE_SIZES,
        uniform_std["GMM 7"]["a"],
        label="U(0,1): GMM 7, a"
    )

    ax.plot(
        SAMPLE_SIZES,
        exponential_std["МО"],
        label="Exp: МО"
    )

    ax.plot(
        SAMPLE_SIZES,
        exponential_std["GMM 3"],
        label="Exp: GMM 3"
    )

    ax.plot(
        SAMPLE_SIZES,
        exponential_std["GMM 5"],
        label="Exp: GMM 5"
    )

    ax.plot(
        SAMPLE_SIZES,
        exponential_std["GMM 7"],
        label="Exp: GMM 7"
    )

    ax.set_title(
        "Сравнение СКО различных оценок"
    )

    ax.set_xlabel("Размер выборки n")
    ax.set_ylabel("СКО")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figures.append(
        (fig, "8 / 8 — Сравнение СКО")
    )

    return figures


# ============================================================
# ПЕРЕКЛЮЧЕНИЕ ГРАФИКОВ СТРЕЛКАМИ
# ============================================================

# ============================================================
# ПЕРЕКЛЮЧАТЕЛЬ ГРАФИКОВ
# ============================================================

class GraphViewer:
    def __init__(self, figures):
        self.current = 0
        self.pages = []

        # Сохраняем каждый график как изображение,
        # после чего закрываем его исходное окно
        for fig, title in figures:
            fig.canvas.draw()

            image = np.asarray(fig.canvas.buffer_rgba()).copy()

            self.pages.append({
                "image": image,
                "title": title
            })

            plt.close(fig)

        # Создаём ОДНО окно для просмотра
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        try:
            self.fig.canvas.manager.set_window_title(
                "Лабораторная работа №3 — графики"
            )
        except AttributeError:
            pass

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        self.show_current()

    def show_current(self):
        self.ax.clear()

        page = self.pages[self.current]

        self.ax.imshow(page["image"])
        self.ax.axis("off")

        self.fig.suptitle(
            page["title"],
            fontsize=12
        )

        self.fig.text(
            0.5,
            0.02,
            f"← / → — переключение    Esc — выход    "
            f"График {self.current + 1} из {len(self.pages)}",
            ha="center",
            fontsize=10
        )

        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        self.fig.canvas.draw_idle()

    def on_key(self, event):

        if event.key == "right":
            self.current = (
                self.current + 1
            ) % len(self.pages)

            self.show_current()

        elif event.key == "left":
            self.current = (
                self.current - 1
            ) % len(self.pages)

            self.show_current()

        elif event.key == "escape":
            plt.close(self.fig)
# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================
# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Равномерное — классический метод моментов
    # --------------------------------------------------------

    print("[1/6] Равномерное: классический МО...")

    uniform_classic = calculate_uniform_classic()


    # --------------------------------------------------------
    # 2. Равномерное — GMM
    # --------------------------------------------------------

    print("[2/6] Равномерное: GMM...")

    uniform_gmm = calculate_uniform_gmm()


    # --------------------------------------------------------
    # 3. Равномерное — СКО
    # --------------------------------------------------------

    print("[3/6] Равномерное: расчёт СКО...")

    uniform_std = calculate_uniform_std()


    # --------------------------------------------------------
    # 4. Экспоненциальное — классический МО
    # --------------------------------------------------------

    print("[4/6] Экспоненциальное: классический МО...")

    exponential_classic = calculate_exponential_classic()


    # --------------------------------------------------------
    # 5. Экспоненциальное — GMM
    # --------------------------------------------------------

    print("[5/6] Экспоненциальное: GMM...")

    exponential_gmm = calculate_exponential_gmm()


    # --------------------------------------------------------
    # 6. Экспоненциальное — СКО
    # --------------------------------------------------------

    print("[6/6] Экспоненциальное: расчёт СКО...")

    exponential_std = calculate_exponential_std()


    # --------------------------------------------------------
    # Построение графиков
    # --------------------------------------------------------

    print("Построение графиков...")

    figures = create_figures(
        uniform_classic,
        uniform_gmm,
        uniform_std,
        exponential_classic,
        exponential_gmm,
        exponential_std
    )


    print("Готово!")
    print("Используйте ← и → для переключения графиков.")
    print("Esc — выход.")


    # --------------------------------------------------------
    # Просмотр графиков в одном окне
    # --------------------------------------------------------

    viewer = GraphViewer(figures)

    plt.show()


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()
