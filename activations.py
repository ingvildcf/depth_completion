import numpy as np
import matplotlib.pyplot as plt
import math

def relu(x) :
    return max(x, 0)

def elu(x, alpha):
    if x>0:
        return x
    else:
        return alpha*(math.exp(x)-1)

x = np.linspace(-5,5,100)

plt.figure(figsize=(12,8))
plt.plot(x, list(map(lambda x: relu(x),x)), label="ReLU", linewidth = 4)
plt.plot(x, list(map(lambda x: elu(x, 1.0),x)), label="ELU, alpha = 1", linestyle = '-.', linewidth = 4)
plt.plot(x, list(map(lambda x: elu(x, 2.0),x)), label="ELU, alpha = 2", linestyle = '--', linewidth = 4, color = 'm')
plt.title("ReLU vs ELU")
plt.legend()
plt.show()