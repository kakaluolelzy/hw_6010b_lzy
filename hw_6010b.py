import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
SIGMA = np.mat('1,0.3,-0.2,0.4; 0.3,1,-0.3,0.1; -0.2,-0.3,1,0.5; 0.4,0.1,0.5,1')
d = 4
r = 0.02
T = 0.5
sigma_1, sigma_2, sigma_3 = 0.1, 0.1, 0.1
sigma_4 = 0.2
sigma_d = np.array([sigma_1, sigma_2, sigma_3, sigma_4])
sample_sizes = np.array([2500, 5000, 10000, 20000, 40000, 80000])
strike_prices = np.arange(50, 81, 5)
initial_prices = np.array([45, 50, 45, 55])


def cholesky_factorization(covariance_matrix=SIGMA):
    a = covariance_matrix
    c = np.zeros(a.shape)
    c[0,0] = a[0, 0]
    for i in range(1, a.shape[0]):
        for j in range(i+1):
            if i > j:
                c[i, j] = (a[i, j] - np.sum((c[i, :]*c[j, :])[:j]))/c[j, j]
            else:
                c[i, j] = (a[i, j] - np.sum((c[i, :] * c[j, :])[:j]))**0.5
    return c


A = cholesky_factorization(SIGMA)


def element_2():
    np.random.seed(123)
    V_estimate_l = []
    V_SE_l = []
    K = strike_prices[0]
    for n in sample_sizes:
        H_l = []
        for i in trange(n):
            Z = np.random.normal(0, 1, d)
            Y = A.dot(Z)
            S_T = initial_prices * np.exp((r - 0.5 * sigma_d**2) * T + sigma_d * T**0.5 * Y)
            H = np.exp(-r * T) * max(0, max(S_T) - K)
            H_l.append(H)
        V_estimate = np.mean(H_l)
        V_SE = np.std(H_l, ddof=1) / n**0.5
        V_estimate_l.append(V_estimate)
        V_SE_l.append(V_SE)
    print(V_estimate_l, V_SE_l)
    plt.plot(np.log(sample_sizes), V_SE_l)
    plt.savefig('element2.png')
    plt.close()

def element_3():
    np.random.seed(123)
    V_estimate_l = []
    V_SE_l = []
    n = sample_sizes[-1]
    for K in strike_prices:
        H_l = []
        for i in trange(n):
            Z = np.random.normal(0, 1, d)
            Y = A.dot(Z)
            S_T = initial_prices * np.exp((r - 0.5 * sigma_d**2) * T + sigma_d * T**0.5 * Y)
            H = np.exp(-r * T) * max(0, max(S_T) - K)
            H_l.append(H)
        V_estimate = np.mean(H_l)
        V_SE = np.std(H_l, ddof=1) / n**0.5
        V_estimate_l.append(V_estimate)
        V_SE_l.append(V_SE)
    print(V_estimate_l, V_SE_l)
    relativa_error = np.array(V_SE_l) / np.array(V_estimate_l)
    plt.plot(strike_prices, relativa_error)
    plt.savefig('element3.png')
    plt.close()


def inverse_transform():
    np.random.seed(123)
    K = strike_prices[0]
    n = sample_sizes[2]
    H_l = []
    for i in trange(n):
        u = np.random.uniform(0, 1, d)
        Z = st.norm.ppf(u)
        Y = A.dot(Z)
        S_T = initial_prices * np.exp((r - 0.5 * sigma_d**2) * T + sigma_d * T**0.5 * Y)
        H = np.exp(-r * T) * max(0, max(S_T) - K)
        H_l.append(H)
    V_estimate = np.mean(H_l)
    V_SE = np.std(H_l, ddof=1) / n**0.5
    return V_estimate, V_SE


def acceptance_rejection():
    """
    assume that -10<= x <= 10, so that uniform distribution could be used as g(x).
    g(x) = 0.05, c = (2 * pai)**-0.5 / 0.05
    """
    np.random.seed(123)
    K = strike_prices[0]
    n = sample_sizes[2]
    H_l = []
    g = 0.05
    c = (2 * np.pi)**-0.5 / 0.05
    for i in trange(n):
        Z = np.zeros(d)
        for j in range(d):
            while True:
                X = np.random.uniform(-10, 10)
                u = np.random.uniform(0, 1)
                if u <= st.norm.pdf(X) / (c * g):
                    Z[j] = X
                    break
                else:
                    continue
        Y = A.dot(Z)
        S_T = initial_prices * np.exp((r - 0.5 * sigma_d ** 2) * T + sigma_d * T ** 0.5 * Y)
        H = np.exp(-r * T) * max(0, max(S_T) - K)
        H_l.append(H)
    V_estimate = np.mean(H_l)
    V_SE = np.std(H_l, ddof=1) / n ** 0.5
    return V_estimate, V_SE


def box_muller():
    np.random.seed(123)
    K = strike_prices[0]
    n = sample_sizes[2]
    H_l = []
    for i in trange(n):
        u = np.random.uniform(0, 1, d)
        Z = np.zeros(d)
        Z[0] = (-2 * np.log(u[0]))**0.5 * np.cos(2 * np.pi * u[1])
        Z[1] = (-2 * np.log(u[0]))**0.5 * np.sin(2 * np.pi * u[1])
        Z[2] = (-2 * np.log(u[2])) ** 0.5 * np.cos(2 * np.pi * u[3])
        Z[3] = (-2 * np.log(u[2])) ** 0.5 * np.sin(2 * np.pi * u[3])
        Y = A.dot(Z)
        S_T = initial_prices * np.exp((r - 0.5 * sigma_d ** 2) * T + sigma_d * T ** 0.5 * Y)
        H = np.exp(-r * T) * max(0, max(S_T) - K)
        H_l.append(H)
    V_estimate = np.mean(H_l)
    V_SE = np.std(H_l, ddof=1) / n ** 0.5
    return V_estimate, V_SE


def element_4():
    results = {}
    results['inve'] = inverse_transform()
    results['acce'] = acceptance_rejection()
    results['boxm'] = box_muller()
    dt = pd.DataFrame(results, index=['mean', 'sd'])
    dt.to_csv('comparison.csv')





