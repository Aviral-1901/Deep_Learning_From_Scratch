import numpy as np

def sigmoid(Z) : 
    return 1 / (1 + np.exp(-Z))

X = np.array([[2,3]])
y = np.array([1])
lr = 0.1

W1 = np.random.randn(2,3) * 0.1
B1 = np.zeros((1,3))
W2 = np.random.randn(3,1) * 0.1
B2 = np.zeros((1,1))
Y = np.array([[1]])

for i in range(50):
    #forward pass
    Z1 = X @ W1 + B1
    OUT1 = sigmoid(Z1)
    Z2 = OUT1 @ W2 + B2
    LOSS = (Z2 - Y)**2

    #backward pass
    dL_dOut2 = 2 * (Z2 - Y)
    dL_dW2 = OUT1.T @ dL_dOut2 
    dL_dB2 = dL_dOut2
    dL_dOut1 = dL_dOut2 @ W2.T
    dL_dZ1 = dL_dOut1 * OUT1 * (1 - OUT1)
    dL_dW1 = X.T @ dL_dZ1
    dL_dB1 = dL_dZ1

    W2 = W2 - lr * dL_dW2
    B2 = B2 - lr * dL_dB2
    W1 = W1 - lr * dL_dW1
    B1 = B1 - lr * dL_dB1
    if i%5 ==0 :
            print(f"Iteration {i} | Loss: {LOSS.item()}")

print("W1 is",W1)
print("W2 is",W2)


