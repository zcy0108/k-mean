import numpy as np
import matplotlib.pyplot as plt

for k in range(10):
    a = np.random.randint(100, size=100)
    x = [i for i in range(100)]
    y = list()
    for i in range(100):
        y.append(np.abs(np.sum(i - a)))
    y = np.array(y)
    np.sort(y)
    print(np.sum(a)/100, np.argmin(y), )
