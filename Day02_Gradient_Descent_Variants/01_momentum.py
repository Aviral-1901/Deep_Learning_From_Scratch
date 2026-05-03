import numpy as np

def sigmoid(Z) : 
    return 1 / (1 + np.exp(-Z))

X = np.array([[2,3]])
Y = np.array([1])
lr = 0.1
beta = 0.9

W1 = np.random.randn(2,3) * 0.1
B1 = np.zeros((1,3))
W2 = np.random.randn(3,1) * 0.1
B2 = np.zeros((1,1))

#velocity terms
vW1 = np.zeros((2,3))
vB1 = np.zeros((1,3))
vW2 = np.zeros((3,1))
vB2 = np.zeros((1,1))

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

    #momentum update
    vW2 = beta * vW2 + (1 - beta) * dL_dW2
    W2 = W2 - lr * vW2

    vB2 = beta * vB2 + (1 - beta) * dL_dB2
    B2 = B2 - lr * vB2

    vW1 = beta * vW1 + (1 - beta) * dL_dW1
    W1 = W1 - lr * vW1

    vB1 = beta * vB1 + (1 - beta) * dL_dB1
    B1 = B1 - lr * vB1


    if i%5 ==0 :
            print(f"Iteration {i} | Loss: {LOSS.item()}")

print("W1 is",W1)
print("W2 is",W2)


