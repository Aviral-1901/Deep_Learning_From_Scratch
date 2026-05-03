import numpy as np

def sigmoid(Z) : 
    return 1 / (1 + np.exp(-Z))


X = np.array([[2,3]])
Y = np.array([1])

lr = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

W1 = np.random.randn(2,3) * 0.1
B1 = np.zeros((1,3))
W2 = np.random.randn(3,1) * 0.1
B2 = np.zeros((1,1))

#v and m terms
vW1 = np.zeros((2,3))
vB1 = np.zeros((1,3))
vW2 = np.zeros((3,1))
vB2 = np.zeros((1,1))
mW1 = np.zeros((2,3))
mB1 = np.zeros((1,3))
mW2 = np.zeros((3,1))
mB2 = np.zeros((1,1))

t = 1

for i in range(200):
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

    #adam update
    mW2 = beta1 * mW2 + (1 - beta1) * dL_dW2       # momentum term
    vW2 = beta2 * vW2 + (1 - beta2) * dL_dW2**2       # squared gradient term
    m_correctedW2 = mW2 / (1 - beta1**t)               # bias correction
    v_correctedW2 = vW2 / (1 - beta2**t)               # bias correction
    W2 = W2 - lr * m_correctedW2 / (np.sqrt(v_correctedW2) + eps)

    mB2 = beta1 * mB2 + (1 - beta1) * dL_dB2      # momentum term
    vB2 = beta2 * vB2 + (1 - beta2) * dL_dB2**2       # squared gradient term
    m_correctedB2 = mB2 / (1 - beta1**t)               # bias correction
    v_correctedB2 = vB2 / (1 - beta2**t)               # bias correction
    B2 = B2 - lr * m_correctedB2 / (np.sqrt(v_correctedB2) + eps)

    mW1 = beta1 * mW1 + (1 - beta1) * dL_dW1       # momentum term
    vW1 = beta2 * vW1 + (1 - beta2) * dL_dW1**2       # squared gradient term
    m_correctedW1 = mW1 / (1 - beta1**t)               # bias correction
    v_correctedW1 = vW1 / (1 - beta2**t)               # bias correction
    W1 = W1 - lr * m_correctedW1 / (np.sqrt(v_correctedW1) + eps)

    mB1 = beta1 * mB1 + (1 - beta1) * dL_dB1      # momentum term
    vB1 = beta2 * vB1 + (1 - beta2) * dL_dB1**2       # squared gradient term
    m_correctedB1 = mB1 / (1 - beta1**t)               # bias correction
    v_correctedB1 = vB1 / (1 - beta2**t)               # bias correction
    B1 = B1 - lr * m_correctedB1 / (np.sqrt(v_correctedB1) + eps)

    t += 1

    if i%5 ==0 :
            print(f"Iteration {i} | Loss: {LOSS.item()}")

print("W1 is",W1)
print("W2 is",W2)


