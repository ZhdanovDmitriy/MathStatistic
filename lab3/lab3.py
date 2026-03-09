import numpy as np
import matplotlib.pyplot as plt

def outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    return data[(data < low) | (data > high)]

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
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), n)


distributions = ["normal", "cauchy", "laplace", "poisson", "uniform"]
sizes = [20, 100]

for n in sizes:

    data = [generate(d, n) for d in distributions]

    plt.figure()
    plt.boxplot(data)
    plt.xticks(range(1,6), distributions)
    plt.title(f"Боксплот, n={n}")
    plt.show()


for dist in distributions:
    for n in sizes:

        shares = []

        for i in range(1000):

            sample = generate(dist, n)

            outs = outliers(sample)

            share = len(outs) / n

            shares.append(share)

        mean_share = np.mean(shares)

        print(dist, "n=", n, "средняя доля выбросов =", mean_share)