import networkx as nx
from scipy.stats import zipfian, poisson, randint, binom
import numpy as np
from sampling_model import sample
from experiment import experiment
import pickle

# Path to the results directory
results_dir = 'results/'

# Parameter for the zipf distribution
a = 1.132

# Parameter for the poisson distribution
lambda_poisson = 3

# Definition of the truncated poisson distribution
def truncated_poisson_pmf(min_value, max_value, mu, k):
    min_value_poisson = poisson.cdf(min_value, mu)
    max_value_poisson = poisson.cdf(max_value, mu)
    return ((mu**k)*(np.e**(-mu)))/(np.math.factorial(k)*(max_value_poisson - min_value_poisson))

# Definition of the experiment.
experiment_sets =[
    ([0.2, 0.5, 0.3], [randint.pmf(i, 1, 4) for i in range(1, 4)], 3, 3, 0.2, 20000),
    ([0.2, 0.5, 0.3], [randint.pmf(i, 1, 4) for i in range(1, 4)], 3, 3, 0.5, 20000),
    ([0.2, 0.5, 0.3], [randint.pmf(i, 1, 4) for i in range(1, 4)], 3, 3, 0.8, 20000),
    ([truncated_poisson_pmf(0, 11, i, lambda_poisson) for i in range(1, 11)], [binom.pmf(i, 9, 0.3) for i in range(10)], 10, 3, 0.2, 20000),
    ([truncated_poisson_pmf(0, 11, i, lambda_poisson) for i in range(1, 11)], [binom.pmf(i, 9, 0.3) for i in range(10)], 10, 3, 0.5, 20000),
    ([truncated_poisson_pmf(0, 11, i, lambda_poisson) for i in range(1, 11)], [binom.pmf(i, 9, 0.3) for i in range(10)], 10, 3, 0.8, 20000),
    ([zipfian.pmf(i, a, 10) for i in range(1, 11)], [binom.pmf(i, 9, 0.3) for i in range(10)], 10, 3, 0.2, 20000),
    ([zipfian.pmf(i, a, 10) for i in range(1, 11)], [binom.pmf(i, 9, 0.3) for i in range(10)], 10, 3, 0.5, 20000),
    ([zipfian.pmf(i, a, 10) for i in range(1, 11)], [binom.pmf(i, 9, 0.3) for i in range(10)], 10, 3, 0.8, 20000)
]

# -------------------------------------------------------------------------
# Run the experiment
# -------------------------------------------------------------------------
results = {}
for i, exp in enumerate(experiment_sets):
    print(i)

    offspring_distribution, dist_g, W_e, L_e, p, n_steps = exp

    result = experiment(offspring_distribution, dist_g, W_e, L_e, p, n_steps)

    results[i] = {
        'kl_divergence': result[0],
        'estimated_distribution': result[1]
    }

with open(results_dir+'experiment_results_3.pickle', 'wb') as file:
    pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

