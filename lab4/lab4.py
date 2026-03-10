import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# эмпирическая функция распределения
def ecdf(sample):
    x = np.sort(sample)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


sizes = [20, 60, 100]

# список непрерывных распределений
distributions = [

    ("Нормальное N(0,1)",
     lambda n: stats.norm.rvs(size=n),
     lambda x: stats.norm.cdf(x),
     lambda x: stats.norm.pdf(x),
     -4, 4),

    ("Коши C(0,1)",
     lambda n: stats.cauchy.rvs(size=n),
     lambda x: stats.cauchy.cdf(x),
     lambda x: stats.cauchy.pdf(x),
     -4, 4),

    ("Лапласа L(0,1/√2)",
     lambda n: stats.laplace.rvs(scale=1/np.sqrt(2), size=n),
     lambda x: stats.laplace.cdf(x, scale=1/np.sqrt(2)),
     lambda x: stats.laplace.pdf(x, scale=1/np.sqrt(2)),
     -4, 4),

    ("Равномерное U(-√3, √3)",
     lambda n: stats.uniform.rvs(loc=-np.sqrt(3), scale=2*np.sqrt(3), size=n),
     lambda x: stats.uniform.cdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3)),
     lambda x: stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3)),
     -4, 4)
]


# непрерывные распределения
for name, sampler, cdf, pdf, left, right in distributions:

    for n in sizes:

        sample = sampler(n)

        x = np.linspace(left, right, 1000)

        x_ecdf, y_ecdf = ecdf(sample)

        plt.figure()
        plt.plot(x, cdf(x), label="Теоретическая функция распределения")
        plt.step(x_ecdf, y_ecdf, where="post",
                 label="Эмпирическая функция распределения")
        plt.title(f"{name}, n={n}")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure()

        plt.hist(sample, bins=15, density=True,
                 alpha=0.5, label="Гистограмма")

        sns.kdeplot(sample, bw_adjust=1,
                    label="Ядерная оценка плотности")

        plt.plot(x, pdf(x),
                 label="Теоретическая плотность вероятности")

        plt.title(f"{name}, n={n}")
        plt.legend()
        plt.grid()
        plt.show()


for n in sizes:

    sample = stats.poisson.rvs(10, size=n)

    x = np.arange(6, 15)

    x_ecdf, y_ecdf = ecdf(sample)

    plt.figure()
    plt.step(x_ecdf, y_ecdf, where="post",
             label="Эмпирическая функция распределения")

    plt.plot(x, stats.poisson.cdf(x, 10),
             label="Теоретическая функция распределения")

    plt.title(f"Распределение Пуассона P(10), n={n}")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()

    plt.hist(sample, bins=range(6, 16), density=True,
             alpha=0.6, label="Гистограмма")

    plt.plot(x, stats.poisson.pmf(x, 10), 'o-',
             label="Теоретическая функция вероятности")

    plt.title(f"Распределение Пуассона P(10), n={n}")
    plt.legend()
    plt.grid()
    plt.show()