# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

random_walk = np.random.randn(100000)
random_walk

fig = plt.figure(figsize=(10, 10))
plt.title("random walk", fontsize=20)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.hist(random_walk, 1000)


