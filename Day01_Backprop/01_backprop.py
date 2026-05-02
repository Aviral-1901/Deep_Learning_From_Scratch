import numpy as np

def sigmoid(z):
      return 1 / (1 + np.exp(-z))

# Values
x  = 2.0
w1 = 0.5
w2 = 0.3
y  = 1.0
lr = 0.1

for i in range(20):

    #forward pass
    hidden_raw = x * w1
    hidden = sigmoid(hidden_raw)
    output = hidden * w2
    loss = (output - y)**2

    #backward pass
    dl_dout = 2 * (output - y)
    dout_dhidden = w2
    dout_dw2 = hidden
    dhidden_dw1 = sigmoid(hidden_raw) * (1-sigmoid(hidden_raw)) * x
    w2_grad = dl_dout * dout_dw2
    w1_grad = dl_dout * dout_dhidden * dhidden_dw1

    w2 = w2 - w2_grad * lr
    w1 = w1 - w1_grad* lr
    if i%5 ==0 :
            print(f"Iteration {i} | Loss: {loss:.6f} | w1: {w1:.4f} | w2: {w2:.4f}")

print("W1 is",w1)
print("W2 is",w2)
