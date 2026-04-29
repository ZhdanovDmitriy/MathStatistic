import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def ecdf(sample):

    x = np.sort(sample)
    y = np.arange(1, len(x) + 1) / len(x)

    x = np.concatenate(([x[0]], x, [x[-1]]))
    y = np.concatenate(([0], y, [1]))

    return x, y


def get_bins_no_gaps(sample, bins, left, right):

    while bins > 1:
        hist, edges = np.histogram(
            sample,
            bins=bins,
            range=(left, right)
        )

        if np.all(hist > 0):
            return bins, edges

        bins -= 1

    # fallback (если вообще никак)
    hist, edges = np.histogram(
        sample,
        bins=1,
        range=(left, right)
    )
    return 1, edges


def hist_cdf(sample, bins, left, right):

    bins, edges = get_bins_no_gaps(sample, bins, left, right)

    hist, edges = np.histogram(
        sample,
        bins=bins,
        range=(left, right),
        density=True
    )

    dx = edges[1] - edges[0]

    cdf = np.cumsum(hist * dx)

    x = edges[1:]

    return x, cdf, bins


def kde_manual(sample, x, h):

    kde = stats.gaussian_kde(sample, bw_method=h)

    return kde(x)


sizes = [20, 60, 100]

bandwidths = [0.3, 0.7, 1.2]


distributions = [

    ("Нормальное распределение",
     lambda n: stats.norm.rvs(size=n),
     lambda x: stats.norm.cdf(x),
     lambda x: stats.norm.pdf(x),
     -4, 4),

    ("Распределение Коши",
     lambda n: stats.cauchy.rvs(size=n),
     lambda x: stats.cauchy.cdf(x),
     lambda x: stats.cauchy.pdf(x),
     -4, 4),

    ("Распределение Лапласа",
     lambda n: stats.laplace.rvs(scale=1/np.sqrt(2), size=n),
     lambda x: stats.laplace.cdf(x, scale=1/np.sqrt(2)),
     lambda x: stats.laplace.pdf(x, scale=1/np.sqrt(2)),
     -4, 4),

    ("Равномерное распределение",
     lambda n: stats.uniform.rvs(
         loc=-np.sqrt(3),
         scale=2*np.sqrt(3),
         size=n
     ),
     lambda x: stats.uniform.cdf(
         x,
         loc=-np.sqrt(3),
         scale=2*np.sqrt(3)
     ),
     lambda x: stats.uniform.pdf(
         x,
         loc=-np.sqrt(3),
         scale=2*np.sqrt(3)
     ),
     -4,
     4)
]


for name, sampler, cdf, pdf, left, right in distributions:

    for n in sizes:

        sample = sampler(n)

        x = np.linspace(left, right, 1000)

        x_ecdf, y_ecdf = ecdf(sample)

        x_hist, cdf_hist, bins_used = hist_cdf(
            sample,
            bins=15,
            left=left,
            right=right
        )

        plt.figure()

        plt.plot(
            x,
            cdf(x),
            label="Теоретическая функция распределения"
        )

        plt.step(
            x_ecdf,
            y_ecdf,
            where="post",
            label="Эмпирическая функция распределения"
        )

        plt.step(
            x_hist,
            cdf_hist,
            where="post",
            label="Функция распределения по гистограмме"
        )

        plt.xlim(left, right)
        plt.ylim(0, 1)

        plt.title(name + ", объем выборки = " + str(n))
        plt.legend()
        plt.grid()
        plt.show()


        bins_used, _ = get_bins_no_gaps(sample, 15, left, right)

        plt.figure()

        plt.hist(
            sample,
            bins=bins_used,
            range=(left, right),
            density=True,
            alpha=0.4,
            label="Гистограмма"
        )

        plt.plot(
            x,
            pdf(x),
            label="Теоретическая плотность вероятности",
            linewidth=2
        )

        for h in bandwidths:

            y_kde = kde_manual(sample, x, h)

            plt.plot(
                x,
                y_kde,
                label="Ядерная оценка плотности, ширина = " + str(h)
            )

        plt.xlim(left, right)

        plt.title(name + ", объем выборки = " + str(n))
        plt.legend()
        plt.grid()
        plt.show()



for n in sizes:

    left = 6
    right = 14

    sample = stats.poisson.rvs(10, size=n)

    x = np.arange(left, right + 1)

    x_ecdf, y_ecdf = ecdf(sample)

    x_hist, cdf_hist, bins_used = hist_cdf(
        sample,
        bins=8,
        left=left,
        right=right
    )

    plt.figure()

    plt.step(
        x_ecdf,
        y_ecdf,
        where="post",
        label="Эмпирическая функция распределения"
    )

    plt.step(
        x_hist,
        cdf_hist,
        where="post",
        label="Функция распределения по гистограмме"
    )

    plt.plot(
        x,
        stats.poisson.cdf(x, 10),
        marker="o",
        label="Теоретическая функция распределения"
    )

    plt.xlim(left, right)
    plt.ylim(0, 1)

    plt.title(
        "Распределение Пуассона, объем выборки = "
        + str(n)
    )

    plt.legend()
    plt.grid()
    plt.show()


    bins_used, _ = get_bins_no_gaps(sample, 8, left, right)

    plt.figure()

    plt.hist(
        sample,
        bins=bins_used,
        range=(left, right),
        density=True,
        alpha=0.5,
        label="Гистограмма"
    )

    plt.plot(
        x,
        stats.poisson.pmf(x, 10),
        "o-",
        label="Теоретическая функция вероятности"
    )

    plt.title(
        "Распределение Пуассона, объем выборки = "
        + str(n)
    )

    plt.legend()
    plt.grid()
    plt.show()