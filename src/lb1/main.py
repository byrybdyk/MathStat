import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sample_sizes = [10, 100, 1000, 10000]

# Чтобы результаты каждый раз были одинаковыми
np.random.seed(42)

# Генерация всех выборок

samples = {}

for n in sample_sizes:
    samples[n] = np.random.uniform(0, 1, n)

current_page = 0


# Математическое вычисление ЭФР
# Fₙ(x) = (1/n) * sum(I(Xᵢ <= x))

def empirical_cdf(sample, x):

    n = len(sample)

    # I(Xᵢ <= x)
    # True = 1, False = 0
    indicators = sample <= x

    # Суммируем индикаторы и делим на размер выборки n
    return np.sum(indicators) / n


# Математическое вычисление гистограммы
# hⱼ = mⱼ / (n * Δx)
#
# mⱼ — количество точек в j-м интервале
# n — размер выборки
# Δx — ширина интервала

def calculate_histogram(sample, bins):

    n = len(sample)

    # Границы интервалов
    bin_edges = np.linspace(0, 1, bins + 1)

    # Ширина одного интервала
    bin_width = bin_edges[1] - bin_edges[0]

    # Список для количества точек в каждом интервале
    counts = []

    # Проходим по всем интервалам
    for i in range(bins):

        # Левая и правая граница текущего интервала
        left = bin_edges[i]
        right = bin_edges[i + 1]

        # Для последнего интервала включаем правую границу
        if i == bins - 1:
            count = np.sum(
                (sample >= left) &
                (sample <= right)
            )
        else:
            count = np.sum(
                (sample >= left) &
                (sample < right)
            )
        # Добавляем количество точек
        # в текущем интервале
        counts.append(count)

    # Преобразуем список в массив
    counts = np.array(counts)

    # Вычисляем высоту каждого столбика
    # hⱼ = mⱼ / (n * Δx)
    heights = counts / (n * bin_width)

    return bin_edges, heights


# Создание окна

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plt.subplots_adjust(bottom=0.18)


def update_page():

    # Очищаем оба графика
    axes[0].clear()
    axes[1].clear()

    # Текущий размер выборки
    n = sample_sizes[current_page]

    # Получаем выборку
    sample = samples[n]

    # ЛЕВЫЙ ГРАФИК — ЭФР

    # Значения x, для которых вычисляем ЭФР
    x = np.linspace(0, 1, 1000)

    # МАТЕМАТИЧЕСКИ вычисляем ЭФР
    y = np.array([
        empirical_cdf(sample, value)
        for value in x
    ])

    # Строим лесенку по рассчитанным значениям ЭФР
    axes[0].step(
        x,
        y,
        where='post',
        label='Эмпирическая функция Fₙ(x)'
    )

    # Теоретическая функция распределения
    # Для U[0,1]: F(x) = x
    x_theoretical = np.linspace(0, 1, 1000)

    axes[0].plot(
        x_theoretical,
        x_theoretical,
        '--',
        label='Теоретическая функция F(x) = x'
    )

    axes[0].set_title(
        f'Эмпирическая функция распределения\nn = {n}'
    )

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('F(x)')

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.05)

    axes[0].grid(True)
    axes[0].legend()

    # ГИСТОГРАММА

    # Количество полос
    bins = n // 10

    bin_edges, heights = calculate_histogram(sample, bins)

    # Ширина одного столбика
    bin_width = bin_edges[1] - bin_edges[0]

    # Строим столбики по рассчитанным высотам
    axes[1].bar(
        bin_edges[:-1],
        heights,
        width=bin_width,
        align='edge',
        edgecolor='black'
    )

    # Теоретическая плотность
    # Для U[0,1]: f(x) = 1
    axes[1].axhline(
        y=1,
        linestyle='--',
        label='Теоретическая плотность f(x) = 1'
    )

    axes[1].set_title(
        f'Гистограмма\nn = {n}, полос = {bins}'
    )

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Плотность')

    axes[1].set_xlim(0, 1)

    axes[1].grid(True)
    axes[1].legend()


    fig.suptitle(
        f'Лабораторная работа №1 — страница '
        f'{current_page + 1} из {len(sample_sizes)}',
        fontsize=14
    )

    fig.canvas.draw_idle()

ax_previous = plt.axes((0.30, 0.04, 0.15, 0.07))

button_previous = Button(
    ax_previous,
    '← Назад'
)


def previous_page(event):

    global current_page

    if current_page > 0:
        current_page -= 1
        update_page()


button_previous.on_clicked(previous_page)


ax_next = plt.axes((0.55, 0.04, 0.15, 0.07))

button_next = Button(
    ax_next,
    'Вперёд →'
)


def next_page(event):

    global current_page

    if current_page < len(sample_sizes) - 1:
        current_page += 1
        update_page()


button_next.on_clicked(next_page)

update_page()

plt.show()
