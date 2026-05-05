import numpy as np
import matplotlib.pyplot as plt

predictions = np.linspace(0.01, 0.99, 200)
y_true = 1

def mse_loss(y_pred):
    return (y_pred - y_true)**2

def cross_entropy_loss(y_pred) :
    return -(y_true * np.log(y_pred + 1e-8) + (1-y_true) * np.log(1 - y_pred + 1e-8))

def mse_gradient(y_pred):
    return 2 * (y_pred - y_true)

def cross_entropy_gradient(y_pred) :
    return y_pred - y_true

mse_losses = mse_loss(predictions)
cross_entropy_losses = cross_entropy_loss(predictions)
mse_grads = mse_gradient(predictions)
ce_grads = cross_entropy_gradient(predictions)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(predictions, mse_losses, label='MSE')
ax.plot(predictions, cross_entropy_losses, label='Cross_Entropy')
ax.set_title('Loss vs prediction')
ax.legend()
ax.grid(True)

ax = axes[1]
ax.plot(predictions, mse_grads, label='MSE-gradient')
ax.plot(predictions, ce_grads, label='Cross Entropy Gradient')
ax.set_title('Gradient vs prediction')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()