import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.1)
y = x

plt.plot(x,y)
plt.savefig('plot.png')
