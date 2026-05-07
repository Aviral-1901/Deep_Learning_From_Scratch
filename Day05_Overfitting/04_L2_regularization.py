import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) #23
def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + 0.1*np.random.randn(len(x))
indices = np.random.permutation(100)
x = x[indices]
y = y[indices]
x_train = x[:10]
y_train = y[:10]
x_validate = x[10:]
y_validate = y[10:]
lr = 0.05

W1 = 0.1 * np.random.randn(1,100)
B1 = np.zeros((1,100))
W2 = 0.1 * np.random.randn(100,1)
B2 = np.zeros((1,1))
training_losses = []
validation_losses = []

lambda_reg = 0.001

for epoch in range(8000):
    total_train_loss = []
    for i in range(len(x_train)):
        x_input = x_train[i].reshape(1,1)
        y_input = y_train[i].reshape(1,1)

        #forward pass
        Z1 = x_input @ W1 + B1
        OUT1 = sigmoid(Z1)
        OUT = OUT1 @ W2 + B2
        L2_penalty = lambda_reg * (np.sum(W1**2) + np.sum(W2**2))
        LOSS = (OUT - y_input)**2 + L2_penalty

        #backward pass
        dL_dOut = 2 * (OUT - y_input)
        dL_dW2 = OUT1.T @ dL_dOut
        dL_dB2 = dL_dOut
        dL_dout1 = dL_dOut @ W2.T
        dL_dZ1 = dL_dout1 * OUT1 * (1 - OUT1)
        dL_dW1 = x_input.T @ dL_dZ1
        dL_dB1 = dL_dZ1

        W2 = W2 - lr * dL_dW2 - lr * 2 * lambda_reg * W2
        B2 = B2 - lr * dL_dB2
        W1 = W1 - lr * dL_dW1 - lr * 2 * lambda_reg * W1
        B1 = B1 - lr * dL_dB1

        total_train_loss.append(LOSS)

    avg_train_loss = np.mean(total_train_loss)

    total_validation_loss = []
    for i in range(len(x_validate)):
        x_val_input = x_validate[i].reshape(1,1)
        y_val_input = y_validate[i].reshape(1,1)
        Z1 = x_val_input @ W1 + B1
        OUT1 = sigmoid(Z1)
        OUT2 = OUT1 @ W2 + B2
        Validation_Loss = (OUT2 - y_val_input)**2

        total_validation_loss.append(Validation_Loss)
    
    avg_validation_loss = np.mean(total_validation_loss)

    training_losses.append(avg_train_loss)
    validation_losses.append(avg_validation_loss)

print(f"Magnitude of Weight W1 is {np.linalg.norm(W1)}")
print(f"Magnitude of Weight W2 is {np.linalg.norm(W2)}")


plt.plot(training_losses[200:], label="Training Losses")
plt.plot(validation_losses[200:], label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("L2 Regularization")
plt.legend()
plt.grid(True)
plt.show()
