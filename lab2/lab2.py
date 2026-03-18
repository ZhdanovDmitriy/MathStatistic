import numpy as np

# Полусумма экстремальных элементов
def z_R(x):
    return (np.min(x) + np.max(x)) / 2

# Полусумма квартилей
def z_Q(x):
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    return (q1 + q3) / 2

# Усечённое среднее
def z_tr(x):
    n = len(x)
    r = int(0.1 * n)
    xs = np.sort(x)
    trimmed = xs[r:n-r]
    return np.mean(trimmed)


def generate(dist, n):

    if dist == "normal":
        return np.random.normal(0, 1, n)

    if dist == "cauchy":
        return np.random.standard_cauchy(n)

    if dist == "laplace":
        return np.random.laplace(0, 1/np.sqrt(2), n)

    if dist == "poisson":
        return np.random.poisson(10, n)

    if dist == "uniform":
        a = -np.sqrt(3)
        b = np.sqrt(3)
        return np.random.uniform(a, b, n)

def experiment(dist, n):

    mean_vals = []
    med_vals = []
    zr_vals = []
    zq_vals = []
    ztr_vals = []

    for _ in range(1000):

        x = generate(dist, n)

        mean_vals.append(np.mean(x))
        med_vals.append(np.median(x))
        zr_vals.append(z_R(x))
        zq_vals.append(z_Q(x))
        ztr_vals.append(z_tr(x))

    return {
        "Выборочное среднее": (np.mean(mean_vals), np.var(mean_vals)),
        "Медиана": (np.mean(med_vals), np.var(med_vals)),
        "Полусумму экстремальных элементов": (np.mean(zr_vals), np.var(zr_vals)),
        "Полусумму квартилей": (np.mean(zq_vals), np.var(zq_vals)),
        "Усечённое среднее": (np.mean(ztr_vals), np.var(ztr_vals))
    }


distributions = ["normal", "cauchy", "laplace", "poisson", "uniform"]
distributions_ru = ["Нормальное", "Коши", "Лаплас", "Пуасон", "Равномерное"]
sizes = [10, 100, 1000]

for i in range(len(distributions)):
    print(f"\nРаспределение: {distributions_ru[i]}")

    for n in sizes:
        res = experiment(distributions[i], n)
        print(f"    Размер выборки: {n}")

        for key in res:
            m, v = res[key]
            print(f"        {key:<40} E = {m:>8.3f}    D = {v:>8.3f}")