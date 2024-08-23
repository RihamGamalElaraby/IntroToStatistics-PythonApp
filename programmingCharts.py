import random
from collections import Counter
from math import sqrt

import matplotlib.pyplot as plt

# Histogram for a list of data
data = [2, 3, 4, 7, 8, 9, 5, 6, 2, 6, 2]
plt.hist(data, bins=5, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

# Height and Weight data
Height = [
    65.78, 71.52, 69.4, 68.22, 67.79, 68.7, 69.8, 70.01, 67.9, 66.78,
    66.49, 67.62, 68.3, 67.12, 68.28, 71.09, 66.46, 68.65, 71.23, 67.13,
    67.83, 68.88, 63.48, 68.42, 67.63, 67.21, 70.84, 67.49, 66.53, 65.44,
    69.52, 65.81, 67.82, 70.6, 71.8, 69.21, 66.8, 67.66, 67.81, 64.05,
    68.57, 65.18, 69.66, 67.97, 65.98, 68.67, 66.88, 67.7, 69.82, 69.09
]

Weight = [
    112.99, 136.49, 153.03, 142.34, 144.3, 123.3, 141.49, 136.46, 112.37,
    120.67, 127.45, 114.14, 125.61, 122.46, 116.09, 140.0, 129.5, 142.97,
    137.9, 124.04, 141.28, 143.54, 97.9, 129.5, 141.85, 129.72, 142.42,
    131.55, 108.33, 113.89, 103.3, 120.75, 125.79, 136.22, 140.1, 128.75,
    141.8, 121.23, 131.35, 106.71, 124.36, 124.86, 139.67, 137.37, 106.45,
    128.76, 145.68, 116.82, 143.62, 134.93
]

# Histogram for Height
plt.hist(Height)
plt.show()

# Histogram for Weight
plt.hist(Weight)
plt.show()

# Scatter plot for Height vs. Weight
plt.scatter(Height, Weight)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of Height vs. Weight')
plt.show()

# Pie chart for Height
plt.pie(Height)
plt.title('Pie Chart of Height')
plt.show()

# Bar chart for Height vs. Weight
plt.bar(Height, Weight)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Bar Chart of Height vs. Weight')
plt.show()

# Probability functions

# Return the probability of the inverse event (i.e., 1-p)
def inverse_probability(p):
    return 1 - p

print(inverse_probability(0.85))

# Return the probability of two flips resulting in two heads
def two_heads_probability(p):
    return p * p

print(two_heads_probability(0.5))

# Return the probability of getting exactly one head in three flips
def one_head_in_three(p):
    notP = 1 - p
    return p * notP * notP + notP * p * notP + notP * notP * p

print(one_head_in_three(0.6))

# Return the probability of flipping one head each from two coins
def one_head_two_coins(p1, p2):
    return p1 * (1 - p2) + (1 - p1) * p2

print(one_head_two_coins(0.07, 0.34))

# Return the probability of a flip landing on heads with two coins
def coin_flip_probability(p0, p1, p2):
    return p0 * p1 + (1 - p0) * p2

print(coin_flip_probability(0.5, 0.7, 0.3))

# Return the probability of flipping heads given the information
def conditional_probability(p0, p1, p2):
    pB = p1 * p0 + (1 - p2) * (1 - p0)
    return (p1 * p0) / pB

print(conditional_probability(0.3, 0.7, 0.4))

# Return the probability of A conditioned on Not B given P(A) = p0, P(B|A) = p1, and P(Not B|Not A) = p2
def probability_conditioned_on_notB(p0, p1, p2):
    # Insert your code here
    pass  # Add your implementation here

# Flip Predictor Class
class FlipPredictor(object):
    def __init__(self, coins):
        self.coins = coins
        n = len(coins)
        self.probs = [1 / n] * n

    def pheads(self):
        return sum(p * c for p, c in zip(self.probs, self.coins))

    def update(self, result):
        new_probs = []
        for i, coin_prob in enumerate(self.coins):
            if result == 'H':
                new_prob = self.probs[i] * coin_prob
            else:
                new_prob = self.probs[i] * (1 - coin_prob)
            new_probs.append(new_prob)
        total = sum(new_probs)
        self.probs = [p / total for p in new_probs]

# Test the FlipPredictor implementation
def test(coins, flips):
    f = FlipPredictor(coins)
    guesses = []
    for flip in flips:
        f.update(flip)
        guesses.append(f.pheads())
    return guesses

def maxdiff(l1, l2):
    return max([abs(x - y) for x, y in zip(l1, l2)])

testcases = [
    (([0.5, 0.4, 0.3], 'HHTH'), [0.4166666666666667, 0.432, 0.42183098591549295, 0.43639398998330553]),
    (([0.14, 0.32, 0.42, 0.81, 0.21], 'HHHTTTHHH'), [0.5255789473684211, 0.6512136991788505, 0.7295055220497553, 0.6187139453483192, 0.4823974597714815, 0.3895729901052968, 0.46081730193074644, 0.5444108434105802, 0.6297110187222278]),
    (([0.14, 0.32, 0.42, 0.81, 0.21], 'TTTHHHHHH'), [0.2907741935483871, 0.25157009005730924, 0.23136284577678012, 0.2766575695593804, 0.3296000585271367, 0.38957299010529806, 0.4608173019307465, 0.5444108434105804, 0.6297110187222278]),
    (([0.12, 0.45, 0.23, 0.99, 0.35, 0.36], 'THHTHTTH'), [0.28514285714285714, 0.3378256513026052, 0.380956725493104, 0.3518717367468537, 0.37500429586037076, 0.36528605387582497, 0.3555106542906013, 0.37479179323540324]),
    (([0.03, 0.32, 0.59, 0.53, 0.55, 0.42, 0.65], 'HHTHTTHTHHT'), [0.528705501618123, 0.5522060353798126, 0.5337142767315369, 0.5521920592821695, 0.5348391689038525, 0.5152373451083692, 0.535385450497415, 0.5168208803156963, 0.5357708613431963, 0.5510509656933194, 0.536055356823069])
]

for inputs, output in testcases:
    if maxdiff(test(*inputs), output) < 0.001:
        print('Correct')
    else:
        print('Incorrect')

# Likelihood function
def likelihood(dist, data):
    prob = 1
    for roll in data:
        prob *= dist[roll]
    return prob

# Testing likelihood function
tests = [
    (({'A': 0.2, 'B': 0.2, 'C': 0.2, 'D': 0.2, 'E': 0.2}, 'ABCEDDECAB'), 1.024e-07),
    (({'A': 0.3, 'B': 0.3, 'C': 0.3, 'D': 0.1}, 'ABCD'), 0.0027),
    (({'A': 0.1, 'B': 0.1, 'C': 0.8}, 'CCCCCC'), 0.262144),
    (({'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}, 'ABCDDCBAABCD'), 3.814697265625e-08)
]

for t, l in tests:
    if abs(likelihood(*t) / l - 1) < 0.01:
        print('Correct')
    else:
        print('Incorrect')

# Statistical Functions

# Mean calculation
def mean(data):
    return sum(data) / len(data)

# Variance calculation
def variance(data):
    mu = mean(data)
    squared_diffs = [(x - mu) ** 2 for x in data]
    return sum(squared_diffs) / len(data)

# Standard Deviation calculation
def stddev(data):
    return sqrt(variance(data))

# Testing the statistical functions
data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
data3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"Mean of data1: {mean(data1)}")
print(f"Variance of data3: {variance(data3)}")
print(f"Standard Deviation of data3: {stddev(data3)}")

# Flip simulation
def flip(n):
    return [random.choice([0, 1]) for _ in range(n)]

print(flip(10))  # Simulate flipping 10 fair coins
