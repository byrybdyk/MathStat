import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# ============================================================
# 1. ММП ДЛЯ РАВНОМЕРНОГО РАСПРЕДЕЛЕНИЯ
# ============================================================

def uniform_mle(sample):
    """
    ММП-оценки параметров a и b
    для равномерного распределения U(a, b).
    """
    a_est = np.min(sample)
    b_est = np.max(sample)

    return a_est, b_est


# ============================================================
# 2. ММП ДЛЯ РАСПРЕДЕЛЕНИЯ ЛАПЛАСА
# ============================================================

def laplace_mle(sample):
    """
    ММП-оценки параметров c и lambda
    для распределения Лапласа.

    c* = медиана выборки
    lambda* = n / sum(|Xi - c*|)
    """
    n = len(sample)

    c_est = np.median(sample)

    lambda_est = n / np.sum(np.abs(sample - c_est))

    return c_est, lambda_est


# ============================================================
# 3. РАСЧЁТ СКО
# ============================================================

def calculate_std(values):
    """
    Выборочное стандартное отклонение полученных оценок.
    """
    return np.std(values, ddof=1)


# ============================================================
# 4. ЭКСПЕРИМЕНТ ДЛЯ РАВНОМЕРНОГО РАСПРЕДЕЛЕНИЯ
# ============================================================

def experiment_uniform(a_true, b_true, sample_sizes, experiments):
    """
    Для каждого размера выборки генерируется заданное
    количество выборок и рассчитываются ММП-оценки a и b.
    Затем вычисляется СКО этих оценок.
    """

    std_a = []
    std_b = []

    for n in sample_sizes:

        estimates_a = []
        estimates_b = []

        for _ in range(experiments):

            # Генерируем выборку размера n
            sample = np.random.uniform(
                a_true,
                b_true,
                n
            )

            # ММП-оценки
            a_est, b_est = uniform_mle(sample)

            estimates_a.append(a_est)
            estimates_b.append(b_est)

        # СКО оценок
        std_a.append(calculate_std(estimates_a))
        std_b.append(calculate_std(estimates_b))

    return np.array(std_a), np.array(std_b)


# ============================================================
# 5. ЭКСПЕРИМЕНТ ДЛЯ РАСПРЕДЕЛЕНИЯ ЛАПЛАСА
# ============================================================

def experiment_laplace(c_true, lambda_true, sample_sizes, experiments):
    """
    Для каждого размера выборки генерируется заданное
    количество выборок и рассчитываются ММП-оценки c и lambda.
    Затем вычисляется СКО этих оценок.
    """

    std_c = []
    std_lambda = []

    # Для numpy scale = 1 / lambda
    scale = 1 / lambda_true

    for n in sample_sizes:

        estimates_c = []
        estimates_lambda = []

        for _ in range(experiments):

            # Генерируем выборку размера n
            sample = np.random.laplace(
                c_true,
                scale,
                n
            )

            # ММП-оценки
            c_est, lambda_est = laplace_mle(sample)

            estimates_c.append(c_est)
            estimates_lambda.append(lambda_est)

        # СКО оценок
        std_c.append(calculate_std(estimates_c))
        std_lambda.append(calculate_std(estimates_lambda))

    return np.array(std_c), np.array(std_lambda)


# ============================================================
# 6. ОСНОВНАЯ ПРОГРАММА
# ============================================================

def main():

    # --------------------------------------------------------
    # Фиксируем генератор случайных чисел
    # --------------------------------------------------------

    np.random.seed(42)

    # --------------------------------------------------------
    # ИСТИННЫЕ ПАРАМЕТРЫ
    # --------------------------------------------------------

    # Равномерное распределение
    a_true = 2
    b_true = 8

    # Распределение Лапласа
    c_true = 5
    lambda_true = 2

    # --------------------------------------------------------
    # РАЗМЕРЫ ВЫБОРОК
    # --------------------------------------------------------

    sample_sizes = np.array([
        10,
        50,
        100,
        250,
        500,
        750,
        1000
    ])

    # --------------------------------------------------------
    # КОЛИЧЕСТВО ЭКСПЕРИМЕНТОВ
    # --------------------------------------------------------

    experiments = 1000

    # ========================================================
    # ВЫВОД ИНФОРМАЦИИ
    # ========================================================

    print("=" * 70)
    print("РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ")
    print("=" * 70)

    print(f"Истинное значение a = {a_true}")
    print(f"Истинное значение b = {b_true}")

    print()

    print("=" * 70)
    print("РАСПРЕДЕЛЕНИЕ ЛАПЛАСА")
    print("=" * 70)

    print(f"Истинное значение c      = {c_true}")
    print(f"Истинное значение lambda = {lambda_true}")

    # ========================================================
    # ПРИМЕР РАСЧЁТА ОЦЕНОК
    # ========================================================

    print()
    print("=" * 70)
    print("ПРИМЕР РАСЧЕТА ОЦЕНОК")
    print("=" * 70)

    # Для примера используем самый большой размер выборки
    example_n = 1000

    # --------------------------------------------------------
    # Равномерное распределение
    # --------------------------------------------------------

    sample_uniform = np.random.uniform(
        a_true,
        b_true,
        example_n
    )

    a_est, b_est = uniform_mle(sample_uniform)

    print()
    print("Равномерное распределение:")
    print(f"Размер выборки n = {example_n}")
    print("Выборка:")

    print(np.round(sample_uniform, 4))

    print()
    print(f"a* = {a_est:.6f}")
    print(f"b* = {b_est:.6f}")

    # --------------------------------------------------------
    # Распределение Лапласа
    # --------------------------------------------------------

    sample_laplace = np.random.laplace(
        c_true,
        1 / lambda_true,
        example_n
    )

    c_est, lambda_est = laplace_mle(sample_laplace)

    print()
    print("Распределение Лапласа:")
    print(f"Размер выборки n = {example_n}")
    print("Выборка:")

    print(np.round(sample_laplace, 4))

    print()
    print(f"c*      = {c_est:.6f}")
    print(f"lambda* = {lambda_est:.6f}")

    # ========================================================
    # РАСЧЁТ СКО
    # ========================================================

    print()
    print("=" * 90)
    print("ЗАВИСИМОСТЬ СКО ОТ РАЗМЕРА ВЫБОРКИ")
    print("=" * 90)

    std_a, std_b = experiment_uniform(
        a_true,
        b_true,
        sample_sizes,
        experiments
    )

    std_c, std_lambda = experiment_laplace(
        c_true,
        lambda_true,
        sample_sizes,
        experiments
    )

    # ========================================================
    # ТАБЛИЦА РЕЗУЛЬТАТОВ
    # ========================================================

    print(
        f"{'n':>6}"
        f"{'СКО(a*)':>16}"
        f"{'СКО(b*)':>16}"
        f"{'СКО(c*)':>16}"
        f"{'СКО(lambda*)':>20}"
    )

    print("-" * 90)

    for i in range(len(sample_sizes)):

        print(
            f"{sample_sizes[i]:>6}"
            f"{std_a[i]:>16.6f}"
            f"{std_b[i]:>16.6f}"
            f"{std_c[i]:>16.6f}"
            f"{std_lambda[i]:>20.6f}"
        )

    # ========================================================
    # ИНТЕРАКТИВНЫЕ ГРАФИКИ
    # ========================================================

    # Номер текущего графика:
    #
    # 0 — равномерное распределение
    # 1 — распределение Лапласа
    # 2 — все оценки вместе

    current_graph = 0

    fig, ax = plt.subplots(figsize=(11, 7))

    # --------------------------------------------------------
    # Функция построения графика
    # --------------------------------------------------------

    def draw_graph(graph_number):

        ax.clear()

        if graph_number == 0:

            # -----------------------------------------------
            # График 1
            # -----------------------------------------------

            ax.plot(
                sample_sizes,
                std_a,
                marker='o',
                linewidth=2,
                label='СКО(a*)'
            )

            ax.plot(
                sample_sizes,
                std_b,
                marker='s',
                linewidth=2,
                label='СКО(b*)'
            )

            ax.set_title(
                'Равномерное распределение',
                fontsize=16
            )

            ax.set_ylabel(
                'СКО оценок',
                fontsize=12
            )

            ax.legend()

        elif graph_number == 1:

            # -----------------------------------------------
            # График 2
            # -----------------------------------------------

            ax.plot(
                sample_sizes,
                std_c,
                marker='o',
                linewidth=2,
                label='СКО(c*)'
            )

            ax.plot(
                sample_sizes,
                std_lambda,
                marker='s',
                linewidth=2,
                label='СКО(lambda*)'
            )

            ax.set_title(
                'Распределение Лапласа',
                fontsize=16
            )

            ax.set_ylabel(
                'СКО оценок',
                fontsize=12
            )

            ax.legend()

        elif graph_number == 2:

            # -----------------------------------------------
            # График 3
            # -----------------------------------------------

            ax.plot(
                sample_sizes,
                std_a,
                marker='o',
                linewidth=2,
                label='СКО(a*)'
            )

            ax.plot(
                sample_sizes,
                std_b,
                marker='s',
                linewidth=2,
                label='СКО(b*)'
            )

            ax.plot(
                sample_sizes,
                std_c,
                marker='^',
                linewidth=2,
                label='СКО(c*)'
            )

            ax.plot(
                sample_sizes,
                std_lambda,
                marker='d',
                linewidth=2,
                label='СКО(lambda*)'
            )

            ax.set_title(
                'СКО оценок для двух распределений',
                fontsize=16
            )

            ax.set_ylabel(
                'СКО оценок',
                fontsize=12
            )

            ax.legend()

        # ----------------------------------------------------
        # Общие настройки
        # ----------------------------------------------------

        ax.set_xlabel(
            'Размер выборки n',
            fontsize=12
        )

        ax.set_xticks(sample_sizes)

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        # Оставляем место для кнопок
        fig.subplots_adjust(bottom=0.15)

        fig.canvas.draw_idle()

    # ========================================================
    # ПЕРЕКЛЮЧЕНИЕ ГРАФИКОВ
    # ========================================================

    def next_graph(event):

        nonlocal current_graph

        current_graph += 1

        if current_graph > 2:
            current_graph = 0

        draw_graph(current_graph)

    def previous_graph(event):

        nonlocal current_graph

        current_graph -= 1

        if current_graph < 0:
            current_graph = 2

        draw_graph(current_graph)

    # ========================================================
    # ПЕРЕКЛЮЧЕНИЕ СТРЕЛКАМИ КЛАВИАТУРЫ
    # ========================================================

    def keyboard(event):

        if event.key == 'right':
            next_graph(event)

        elif event.key == 'left':
            previous_graph(event)

    fig.canvas.mpl_connect(
        'key_press_event',
        keyboard
    )

    # ========================================================
    # КНОПКИ
    # ========================================================

    ax_previous = plt.axes([
        0.25,
        0.03,
        0.18,
        0.06
    ])

    ax_next = plt.axes([
        0.57,
        0.03,
        0.18,
        0.06
    ])

    button_previous = Button(
        ax_previous,
        '← Назад'
    )

    button_next = Button(
        ax_next,
        'Вперед →'
    )

    button_previous.on_clicked(
        previous_graph
    )

    button_next.on_clicked(
        next_graph
    )

    # ========================================================
    # ПОКАЗ ПЕРВОГО ГРАФИКА
    # ========================================================

    draw_graph(current_graph)

    plt.show()


# ============================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================

if __name__ == "__main__":
    main()
