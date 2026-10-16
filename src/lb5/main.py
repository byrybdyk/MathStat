import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import stats

# ==================================================
# ЛАБОРАТОРНАЯ РАБОТА №5
# Сравнение статистических свойств точечных оценок
# математического ожидания и дисперсии
# ==================================================

# Параметры моделирования
N = 1000
mu = 0
sigma2 = 1
sigma = np.sqrt(sigma2)

sample_sizes = np.arange(10, 101, 10)

# Для воспроизводимости результатов
rng = np.random.default_rng(42)

# ==================================================
# ЧАСТЬ 1. МОДЕЛИРОВАНИЕ
# ==================================================

results = {
    "n": [],
    "mean_mean": [],
    "var_mean": [],
    "mean_var_biased": [],
    "var_var_biased": [],
    "mean_var_unbiased": [],
    "var_var_unbiased": []
}

for n in sample_sizes:

    # Генерируем N выборок размера n из N(0, 1)
    samples = rng.normal(mu, sigma, size=(N, n))

    # Оценка математического ожидания
    mean_estimates = np.mean(samples, axis=1)

    # Неисправленная оценка дисперсии: деление на n
    var_biased = np.var(samples, axis=1, ddof=0)

    # Исправленная оценка дисперсии: деление на n - 1
    var_unbiased = np.var(samples, axis=1, ddof=1)

    # Математические ожидания оценок
    results["n"].append(n)
    results["mean_mean"].append(np.mean(mean_estimates))
    results["mean_var_biased"].append(np.mean(var_biased))
    results["mean_var_unbiased"].append(np.mean(var_unbiased))

    # Дисперсии оценок
    results["var_mean"].append(np.var(mean_estimates, ddof=0))
    results["var_var_biased"].append(np.var(var_biased, ddof=0))
    results["var_var_unbiased"].append(np.var(var_unbiased, ddof=0))

# Преобразуем результаты в массивы
for key in results:
    results[key] = np.array(results[key])

n_values = results["n"]

# ==================================================
# ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ
# ==================================================

# Выборочное среднее:
# E[X̄] = mu
# Var(X̄) = sigma² / n

theory_mean_mean = np.full(len(n_values), mu)
theory_var_mean = sigma2 / n_values

# Неисправленная оценка дисперсии:
# E[D*] = sigma² * (n - 1) / n
# Var(D*) = 2 * sigma^4 * (n - 1) / n²

theory_mean_var_biased = sigma2 * (n_values - 1) / n_values

theory_var_var_biased = (
    2 * sigma2**2 * (n_values - 1) / n_values**2
)

# Исправленная оценка дисперсии:
# E[D] = sigma²
# Var(D) = 2 * sigma^4 / (n - 1)

theory_mean_var_unbiased = np.full(len(n_values), sigma2)

theory_var_var_unbiased = (
    2 * sigma2**2 / (n_values - 1)
)

# ==================================================
# ВЫВОД ТАБЛИЦ В ТЕРМИНАЛ
# ==================================================

print("\n" + "=" * 100)
print("ЛАБОРАТОРНАЯ РАБОТА №5")
print("N =", N, "| Распределение N(0, 1)")
print("=" * 100)

print("\nТАБЛИЦА 1. Математические ожидания оценок")
print(
    f"{'n':>5}"
    f"{'M[X̄] эксп.':>15}"
    f"{'M[X̄] теор.':>15}"
    f"{'M[D*] эксп.':>15}"
    f"{'M[D*] теор.':>15}"
    f"{'M[D] эксп.':>15}"
    f"{'M[D] теор.':>15}"
)

for i, n in enumerate(n_values):
    print(
        f"{n:5d}"
        f"{results['mean_mean'][i]:15.5f}"
        f"{theory_mean_mean[i]:15.5f}"
        f"{results['mean_var_biased'][i]:15.5f}"
        f"{theory_mean_var_biased[i]:15.5f}"
        f"{results['mean_var_unbiased'][i]:15.5f}"
        f"{theory_mean_var_unbiased[i]:15.5f}"
    )

print("\nТАБЛИЦА 2. Дисперсии оценок")
print(
    f"{'n':>5}"
    f"{'D[X̄] эксп.':>15}"
    f"{'D[X̄] теор.':>15}"
    f"{'D[D*] эксп.':>15}"
    f"{'D[D*] теор.':>15}"
    f"{'D[D] эксп.':>15}"
    f"{'D[D] теор.':>15}"
)

for i, n in enumerate(n_values):
    print(
        f"{n:5d}"
        f"{results['var_mean'][i]:15.5f}"
        f"{theory_var_mean[i]:15.5f}"
        f"{results['var_var_biased'][i]:15.5f}"
        f"{theory_var_var_biased[i]:15.5f}"
        f"{results['var_var_unbiased'][i]:15.5f}"
        f"{theory_var_var_unbiased[i]:15.5f}"
    )

# ==================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ ФУНКЦИЙ РАСПРЕДЕЛЕНИЯ
# ==================================================

cdf_results = {}

for n in [10, 100]:

    samples = rng.normal(mu, sigma, size=(N, n))

    means = np.mean(samples, axis=1)
    d_biased = np.var(samples, axis=1, ddof=0)
    d_unbiased = np.var(samples, axis=1, ddof=1)

    cdf_results[n] = {
        "mean": np.sort(means),
        "biased": np.sort(d_biased),
        "unbiased": np.sort(d_unbiased),
        "ecdf": np.arange(1, N + 1) / N
    }

# ==================================================
# СОЗДАНИЕ СТРАНИЦ С ГРАФИКАМИ
# ==================================================

pages = []

# Страница 1. Математические ожидания
pages.append({
    "type": "statistics",
    "title": "Математические ожидания оценок",
    "plots": [
        {
            "title": "Выборочное среднее",
            "emp": results["mean_mean"],
            "theory": theory_mean_mean,
            "ylabel": "M[X̄]"
        },
        {
            "title": "Неисправленная оценка дисперсии",
            "emp": results["mean_var_biased"],
            "theory": theory_mean_var_biased,
            "ylabel": "M[D*]"
        },
        {
            "title": "Исправленная оценка дисперсии",
            "emp": results["mean_var_unbiased"],
            "theory": theory_mean_var_unbiased,
            "ylabel": "M[D]"
        }
    ]
})

# Страница 2. Дисперсии оценок
pages.append({
    "type": "statistics",
    "title": "Дисперсии оценок",
    "plots": [
        {
            "title": "Дисперсия выборочного среднего",
            "emp": results["var_mean"],
            "theory": theory_var_mean,
            "ylabel": "D[X̄]"
        },
        {
            "title": "Дисперсия неисправленной оценки",
            "emp": results["var_var_biased"],
            "theory": theory_var_var_biased,
            "ylabel": "D[D*]"
        },
        {
            "title": "Дисперсия исправленной оценки",
            "emp": results["var_var_unbiased"],
            "theory": theory_var_var_unbiased,
            "ylabel": "D[D]"
        }
    ]
})

# Страницы 3 и 4. Функции распределения
for n in [10, 100]:
    pages.append({
        "type": "cdf",
        "title": f"Функции распределения, n = {n}",
        "n": n
    })

# ==================================================
# ИНТЕРАКТИВНОЕ ОКНО
# ==================================================

current_page = 0

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

plt.subplots_adjust(
    bottom=0.20,
    top=0.82,
    wspace=0.35
)


def update_page():

    page = pages[current_page]

    for ax in axes:
        ax.clear()

    # ----------------------------------------------
    # ГРАФИКИ МАТЕМАТИЧЕСКИХ ОЖИДАНИЙ И ДИСПЕРСИЙ
    # ----------------------------------------------

    if page["type"] == "statistics":

        for i, plot in enumerate(page["plots"]):

            ax = axes[i]

            ax.plot(
                n_values,
                plot["emp"],
                "o-",
                label="Эксперимент"
            )

            ax.plot(
                n_values,
                plot["theory"],
                "--",
                label="Теория"
            )

            # Истинные параметры для наглядности
            if current_page == 0 and i == 1:
                ax.axhline(
                    sigma2,
                    color="gray",
                    linestyle=":",
                    label="Истинная дисперсия"
                )

            ax.set_title(plot["title"])
            ax.set_xlabel("Размер выборки n")
            ax.set_ylabel(plot["ylabel"])
            ax.grid(True)
            ax.legend(fontsize=8)

    # ----------------------------------------------
    # ФУНКЦИИ РАСПРЕДЕЛЕНИЯ
    # ----------------------------------------------

    else:

        n = page["n"]
        data = cdf_results[n]
        y = data["ecdf"]

        # График 1. Выборочное среднее
        ax = axes[0]
        x_mean = data["mean"]

        ax.step(
            x_mean,
            y,
            where="post",
            label="Эмпирическая ФР"
        )

        x = np.linspace(x_mean.min(), x_mean.max(), 500)

        ax.plot(
            x,
            stats.norm.cdf(
                x,
                loc=mu,
                scale=sigma / np.sqrt(n)
            ),
            "--",
            label="Теоретическая ФР"
        )

        ax.set_title("ФР выборочного среднего")
        ax.set_xlabel("Значение")
        ax.set_ylabel("F(x)")
        ax.grid(True)
        ax.legend(fontsize=8)

        # График 2. Неисправленная дисперсия
        ax = axes[1]
        x_biased = data["biased"]

        ax.step(
            x_biased,
            y,
            where="post",
            label="Эмпирическая ФР"
        )

        x = np.linspace(0, x_biased.max() * 1.05, 500)

        ax.plot(
            x,
            stats.chi2.cdf(
                n * x / sigma2,
                df=n - 1
            ),
            "--",
            label="Теоретическая ФР"
        )

        ax.set_title("ФР неисправленной D*")
        ax.set_xlabel("Значение")
        ax.set_ylabel("F(x)")
        ax.grid(True)
        ax.legend(fontsize=8)

        # График 3. Исправленная дисперсия
        ax = axes[2]
        x_unbiased = data["unbiased"]

        ax.step(
            x_unbiased,
            y,
            where="post",
            label="Эмпирическая ФР"
        )

        x = np.linspace(0, x_unbiased.max() * 1.05, 500)

        ax.plot(
            x,
            stats.chi2.cdf(
                (n - 1) * x / sigma2,
                df=n - 1
            ),
            "--",
            label="Теоретическая ФР"
        )

        ax.set_title("ФР исправленной D")
        ax.set_xlabel("Значение")
        ax.set_ylabel("F(x)")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Лабораторная работа №5\n"
        + page["title"]
        + f" | Страница {current_page + 1} из {len(pages)}",
        fontsize=13
    )

    fig.canvas.draw_idle()



# ==================================================
# КНОПКИ И ПЕРЕКЛЮЧЕНИЕ СТРАНИЦ СТРЕЛКАМИ
# ==================================================

ax_previous = plt.axes((0.30, 0.05, 0.15, 0.07))
button_previous = Button(ax_previous, "← Назад")

ax_next = plt.axes((0.55, 0.05, 0.15, 0.07))
button_next = Button(ax_next, "Вперёд →")


def previous_page(event=None):
    global current_page
    # С первой страницы переходим на последнюю
    current_page = (current_page - 1) % len(pages)
    update_page()


def next_page(event=None):
    global current_page
    # С последней страницы переходим на первую
    current_page = (current_page + 1) % len(pages)
    update_page()


# Обработчики кнопок мыши
button_previous.on_clicked(previous_page)
button_next.on_clicked(next_page)


# Обработчик стрелок клавиатуры
def on_key(event):
    if event.key == "right":
        next_page()
    elif event.key == "left":
        previous_page()


fig.canvas.mpl_connect("key_press_event", on_key)

# Показываем первую страницу
update_page()
plt.show()
