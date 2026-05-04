import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 300)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def gelu(z):
    return z * sigmoid(1.702 * z)


#derivative of the functions
d_sigmoid = sigmoid(z) * (1 - sigmoid(z))
d_tanh = 1 - tanh(z)**2
d_relu = (z > 0).astype(float)
d_gelu = (gelu(z + 0.001) - gelu(z)) / 0.001

# create canvas with two plots side by side 1 row 2 column
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# draw left plot
ax = axes[0]
ax.plot(z, sigmoid(z), label='sigmoid')
ax.plot(z, tanh(z), label='tanh')
ax.plot(z, relu(z), label='relu')
ax.plot(z, gelu(z), label='gelu')
ax.set_title("Activation functions")
ax.legend()
ax.grid(True, alpha=0.3)

# draw derivatives on right plot
ax = axes[1]
ax.plot(z, d_sigmoid, label='d_sigmoid')
ax.plot(z, d_tanh, label='d_tanh')
ax.plot(z, d_relu, label='d_relu')
ax.plot(z, d_gelu, label='d_gelu')
ax.set_title("Derivatives")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()   
plt.show()

