#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# linear line
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
yerror = np.array([1.3, 1.7, 3.5, 4.6, 4.5, 6.5, 6.5, 8.6])

# plot line
plt.plot(x,y, color="black")

# plot errors with + -signs
plt.plot(x,yerror, '+')

# draw error vectors
plt.plot(np.vstack([x,x]), np.vstack([y, yerror]), color="red");

# show plot
plt.show()
