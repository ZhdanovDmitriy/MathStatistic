import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

sizes = [10, 100, 1000]

for n in sizes:
    sample = np.random.normal(0, 1, n)
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample, bins='sturges', density=True)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.title("Normal N(0,1), n = " + str(n))
    plt.show()

for n in sizes:
    sample = stats.cauchy.rvs(0, 1, size=n)
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample, bins='sturges', density=True)
    plt.plot(x, stats.cauchy.pdf(x, 0, 1))
    plt.title("Cauchy C(0,1), n = " + str(n))
    plt.show()

for n in sizes:
    sample = np.random.laplace(0, 1/np.sqrt(2), n)
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample,  bins='sturges', density=True)
    plt.plot(x, stats.laplace.pdf(x, 0, 1/np.sqrt(2)))
    plt.title("Laplace L(0, 1/sqrt(2)), n = " + str(n))
    plt.show()

for n in sizes:
    sample = np.random.poisson(10, n)
    x = np.arange(min(sample), max(sample)+1)
    
    plt.figure()
    plt.hist(sample, bins='sturges', density=True)
    plt.plot(x, stats.poisson.pmf(x, 10))
    plt.title("Poisson P(10), n = " + str(n))
    plt.show()

for n in sizes:
    sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample, bins='sturges', density=True)
    plt.plot(x, stats.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)))
    plt.title("Uniform U(-sqrt(3), sqrt(3)), n = " + str(n))
    plt.show()
