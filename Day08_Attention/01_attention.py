import numpy as np

np.random.seed(42)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
])

WQ = np.random.randn(4,4) * 0.1
WK = np.random.randn(4,4) * 0.1
WV = np.random.randn(4,4) * 0.1

Q = X @ WQ
K = X @ WK
V = X @ WV

scores = Q @ K.T / np.sqrt(4) #4 is the dimension of the vectors used
attention_weights = softmax(scores)
final_output = attention_weights @ V
print(final_output)