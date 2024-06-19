from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from IPython.display import Latex

iterations = 10000
np.random.seed(42)
accepted_samples = []
rejected_samples = []

# YOUR CODE HERE
# f(x) = 2 * x **2
for _ in range(iterations):
    x = np.random.uniform(0.0, 1.0)
    f_x = 2 * x**2
    s = np.random.uniform(0.0, 2.0)
    if s < f_x:
        accepted_samples.append(x)
    else:
        rejected_samples.append(x)
    
# Historgram plots of accepted and rejected samples
x = np.linspace(0, 1, 1000)
y = 2 * x**2
plt.plot(x, y)
plt.show()

plt.hist(accepted_samples, bins=20)
plt.show()

plt.hist(rejected_samples, bins=20)
plt.show()

# YOUR CODE HERE
# rejection ratio = target_funtion / envelope function
ratio = np.random.uniform(0.0, 1.0) / np.random.uniform(0.0, 2.0)
print(ratio)

n = 10000
np.random.seed(42)


# To draw samples from f(x) = 2x^2, invert the cumulative distribution function to F^-1.
def sample_inv_cdf(u: np.array) -> np.array:
    # YOUR CODE HERE
        # F = (2/3) * x^3
        # x = F_inv
        return ((3/2) * u)**(1/3)


# Draw `n` samples from F^-1.
u = np.random.rand(n)
samples = sample_inv_cdf(u)

# Plotting the drawn samples
plt.hist(samples, bins=20)
plt.show()

# YOUR CODE HERE
u = 0.3
f_inv = ((3/2) * u)**(1/3)
print(f_inv)

# Target distribution parameters
weight1 = 0.3
mu1 = 0
sigma1 = 2

weight2 = 0.7
mu2 = 10
sigma2 = 2


# Proposal distribution parameters
mu_prop = 3
sigma_prop = 10

# Define the target function f(x)
def target_func(x: np.array) -> np.array:
    # YOUR CODE HERE
        return np.where(x > 0, (1/100) * np.sqrt(x), 0)

def gaussian(x: np.array, mu: float, sigma: float) -> np.array:
    # YOUR CODE HERE
    normal = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return normal

def mix_of_two_gaussians(
    x: np.array,
    weight1: float,
    mu1: float,
    sigma1: float,
    weight2: float,
    mu2: float,
    sigma2: float,
) -> np.array:
    # YOUR CODE HERE
    p = weight1 * gaussian(x, mu1, sigma1) + weight2 * gaussian(x, mu2, sigma2)
    return p

# Plotting the target distribution, target function and proposal distribution for real valued numbers in x.

# Visualization, x from -20 to 20 in steps of 0.1
x = np.arange(-20, 20, 0.1)

# Target function
target = target_func(x)

# Proposal distribution
proposal = gaussian(x, mu_prop, sigma_prop)

# Creating the target distribution p(x) which is a MoG
mog = mix_of_two_gaussians(x, weight1, mu1, sigma1, weight2, mu2, sigma2)

# Plotting the target function f(x), proposal distribution and the target distribution
plt.plot(x, target, "m-")
plt.plot(x, proposal, "r-")
plt.plot(x, mog, "b-")
plt.show()

it = 1000  # no. of samples
np.random.seed(42)

qz = np.random.normal(mu_prop, sigma_prop, size=it)
# The samples of qz are then used to generate gaussian distributed values
qzValue = gaussian(qz, mu_prop, sigma_prop)
pSamples = mix_of_two_gaussians(qz, weight1, mu1, sigma1, weight2, mu2, sigma2)


# Plotting the samples obtained from a known normal distribution
sns.histplot(qz, bins=20, stat="density", kde=True, element="step")
plt.show()

# YOUR CODE HERE
import scipy.integrate as integrate

def integrand(x):
    return target_func(x) * mix_of_two_gaussians(x, weight1, mu1, sigma1, weight2, mu2, sigma2)

expectation, _ = integrate.quad(integrand, -np.inf, np.inf)
print(expectation)

def MH(target, proposal, proposal_conditional, n_iterations: int):
    samples = []
    n_accepted = 0
    mu = 0
    sigma = 1
    h = 1

    # initial state
    x = np.random.rand()
    proposal_old = target(x)

    for i in range(n_iterations):
        xprime = proposal(x)

        # P(x')
        # P(x') / P(x)
        # q(x|x’), q(x'|x)
        target_new = target(xprime)
        target_old = target(x)
        proposal_new = proposal_conditional(x, xprime)
        proposal_old = proposal_conditional(xprime, x)
        
        # YOUR CODE HERE
        alpha = (target_new * proposal_old) / (target_old * proposal_new)
        alpha = min(1, alpha)

        # Acceptance probability A “accepts” the proposed value x'
        # otherwise rejects it by keeping the current value x
        if np.random.rand() <= alpha:
            x = xprime
            proposal_old = proposal_new
            n_accepted += 1

        if i > 2000:
            samples.append(x)

    return samples, n_accepted

def target(x, mu, sigma):
    # YOUR CODE HERE
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))


def proposal(x, h):
    # sample from q
    # YOUR CODE HERE
    return np.random.uniform(x - h/2, x + h/2)

def proposal_conditional(x, xprime, h):
    # compute q(x|x')
    # YOUR CODE HERE
    if x - h/2 <= xprime <= x + h/2:
        return 1/h
    else: 
        return 0

# Given data
np.random.seed(42)
n_iterations = 50000

# Collect the samples x (new positions) from the target distribution along with the acceptance count, naccept
target_fn = partial(target, mu=0, sigma=1)
proposal_fn = partial(proposal, h=1)
proposal_conditional_fn = partial(proposal_conditional, h=1)

x, naccept = MH(
    target=target_fn,
    proposal=proposal_fn,
    proposal_conditional=proposal_conditional_fn,
    n_iterations=n_iterations,
)

# Plot the histogram of samples drawn from the Target distribution
Ns = [3000, 5000, 8000, n_iterations]
print(f"number of accepted samples = {naccept}")

x_t = np.linspace(-3, 3, 1000)
gauss = target_fn(x_t)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for i, ax in enumerate(fig.axes):
    ax.set_title(f"N={Ns[i]}")
    sns.histplot(ax=ax, data=x[: Ns[i]], element="step", kde=True, stat="density")
    ax.plot(x_t, gauss, color="r")

plt.tight_layout()
plt.show()

def target(
    x: np.array,
    weight1: float,
    mu1: float,
    sigma1: float,
    weight2: float,
    mu2: float,
    sigma2: float,
) -> np.array:
    # YOUR CODE HERE
    p = weight1 * gaussian(x, mu1, sigma1) + weight2 * gaussian(x, mu2, sigma2)
    return p


def proposal(mu: float, sigma: float) -> np.array:
    # sample from q
    # YOUR CODE HERE
    return np.random.normal(mu, sigma)


def proposal_conditional(x, xprime, h):
    # compute q(x|x')
    # YOUR CODE HERE
    if x - h/2 <= xprime <= x + h/2:
        return 1/h
    else: 
        return 0
    
def proposal_conditional(x: np.array, y: float, sigma_prop: float):
    # compute q(x | x')
    # YOUR CODE HERE
    prob = 1 / (sigma_prop * np.sqrt(2 * np.pi)) * np.exp(-(y - x)**2 / (2 * sigma_prop**2))
    return prob

n_iterations = 50000
np.random.seed(42)

# Collect the samples x (new positions) from the target distribution along with the acceptance count, naccept
target_fn = partial(
    target,
    weight1=weight1,
    mu1=mu1,
    sigma1=sigma1,
    weight2=weight2,
    mu2=mu2,
    sigma2=sigma2,
)
proposal_fn = partial(proposal, sigma=sigma_prop)
proposal_conditional_fn = partial(proposal_conditional, sigma_prop=sigma_prop)

x, naccept = MH(
    target=target_fn,
    proposal=proposal_fn,
    proposal_conditional=proposal_conditional_fn,
    n_iterations=n_iterations,
)

# Plot the histogram of samples drawn from the Target distribution
Ns = [3000, 5000, 8000, n_iterations]
print(f"number of accepted samples = {naccept}")

x_t = np.linspace(-10, 20, 1000)
gauss = target_fn(x_t)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for i, ax in enumerate(fig.axes):
    ax.set_title(f"N={Ns[i]}")
    sns.histplot(ax=ax, data=x[: Ns[i]], element="step", kde=True, stat="density")
    ax.plot(x_t, gauss, color="r")
    ax.axis([-10, 20, 0, 0.2])

plt.tight_layout()
plt.show()

# import nbformat
# from nbstripout import strip_output

# # 파일 읽기
# with open('sampling.ipynb', 'r', encoding='utf-8') as f:
#     nb = nbformat.read(f, as_version=4)

# # 출력 제거
# nb_stripped = strip_output(nb, keep_output=False, keep_count=False, keep_id=False)

# # 출력이 제거된 파일 저장
# with open('sampling_stripped.ipynb', 'w', encoding='utf-8') as f:
#     nbformat.write(nb_stripped, f)

# import shutil

# # 파일 다운로드를 위한 경로 설정
# shutil.move('sampling_stripped.ipynb', r'C:\Users\User\Downloads\ex3_sampling\sampling_stripped.ipynb')