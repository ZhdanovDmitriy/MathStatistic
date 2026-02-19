import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

sizes = [100, 1000]
bins = ['fd', 'fd'] #sturges/fd

sample = np.random.normal(0, 1, 10)
x = np.linspace(min(sample), max(sample), 1000)

plt.figure()
plt.hist(sample, bins=4, density=True)
plt.plot(x, stats.norm.pdf(x, 0, 1))
plt.title("Normal N(0,1), n = " + str(10))
plt.show()

for i in range(len(sizes)):
    sample = np.random.normal(0, 1, sizes[i])
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample, bins=bins[i], density=True)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.title("Normal N(0,1), n = " + str(sizes[i]))
    plt.show()

left_boarder = -10
right_boarder = 10
sample = stats.cauchy.rvs(0, 1, size=10)
x = np.linspace(left_boarder, right_boarder, 1000)

plt.figure()
plt.xlim(left_boarder, right_boarder)
plt.hist(sample, bins=4, density=True)
plt.plot(x, stats.cauchy.pdf(x, 0, 1))
plt.title("Cauchy C(0,1), n = " + str(10))
plt.show()

for i in range(len(sizes)):
    left_boarder = -10
    right_boarder = 10
    sample = stats.cauchy.rvs(0, 1, size=sizes[i])
    x = np.linspace(left_boarder, right_boarder, 1000)
    
    plt.figure()
    plt.xlim(left_boarder, right_boarder)
    plt.hist(sample, bins=bins[i], density=True)
    plt.plot(x, stats.cauchy.pdf(x, 0, 1))
    plt.title("Cauchy C(0,1), n = " + str(sizes[i]))
    plt.show()

sample = np.random.laplace(0, 1/np.sqrt(2), 10)
x = np.linspace(min(sample), max(sample), 1000)

plt.figure()
plt.hist(sample,  bins=4, density=True)
plt.plot(x, stats.laplace.pdf(x, 0, 1/np.sqrt(2)))
plt.title("Laplace L(0, 1/sqrt(2)), n = " + str(10))
plt.show()

for i in range(len(sizes)):
    sample = np.random.laplace(0, 1/np.sqrt(2), sizes[i])
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample,  bins=bins[i], density=True)
    plt.plot(x, stats.laplace.pdf(x, 0, 1/np.sqrt(2)))
    plt.title("Laplace L(0, 1/sqrt(2)), n = " + str(sizes[i]))
    plt.show()


sample = np.random.poisson(10, 10)
x = np.arange(min(sample), max(sample)+1)

plt.figure()
plt.hist(sample, bins=3, density=True)
plt.plot(x, stats.poisson.pmf(x, 10))
plt.title("Poisson P(10), n = " + str(10))
plt.show()

for i in range(len(sizes)):
    sample = np.random.poisson(10, sizes[i])
    x = np.arange(min(sample), max(sample)+1)
    
    plt.figure()
    plt.hist(sample, bins=bins[i], density=True)
    plt.plot(x, stats.poisson.pmf(x, 10))
    plt.title("Poisson P(10), n = " + str(sizes[i]))
    plt.show()

sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), 10)
x = np.linspace(min(sample), max(sample), 1000)

plt.figure()
plt.hist(sample, bins=4, density=True)
plt.plot(x, stats.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)))
plt.title("Uniform U(-sqrt(3), sqrt(3)), n = " + str(10))
plt.show()

for i in range(len(sizes)):
    sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), sizes[i])
    x = np.linspace(min(sample), max(sample), 1000)
    
    plt.figure()
    plt.hist(sample, bins=bins[i], density=True)
    plt.plot(x, stats.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)))
    plt.title("Uniform U(-sqrt(3), sqrt(3)), n = " + str(sizes[i]))
    plt.show()
