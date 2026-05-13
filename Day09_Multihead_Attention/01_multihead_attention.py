import numpy as np

np.random.seed(42)

def softmax(x) :
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def single_head_attention(Q, K, V) :
    scores = Q @ K.T
    attention_weights = softmax(scores)
    return attention_weights @ V

X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
])

d_model = 4 #input dimension
d_heads = 2 #number of heads
d_head = d_model // d_heads  #dimension perhead

#head-1 weight matrices
WQ1 = np.random.randn(d_model, d_head) * 0.1
WK1 = np.random.randn(d_model, d_head) * 0.1
WV1 = np.random.randn(d_model, d_head) * 0.1

#head-2 weight matrices
WQ2 = np.random.randn(d_model, d_head) * 0.1
WK2 = np.random.randn(d_model, d_head) * 0.1
WV2 = np.random.randn(d_model, d_head) * 0.1

Q1 = X @ WQ1
K1 = X @ WK1
V1 = X @ WV1

Q2 = X @ WQ2
K2 = X @ WK2
V2 = X @ WV2

# attention for each head
head1_output = single_head_attention(Q1, K1, V1)
head2_output = single_head_attention(Q2, K2, V2)

#concatenate both heads
multi_head_output = np.concatenate([head1_output, head2_output], axis=1)

WO = np.random.randn(d_model, d_model) * 0.1
output = multi_head_output @ WO

print(f"Head 1 output shape: {head1_output.shape}")
print(f"Head 2 output shape: {head2_output.shape}")
print(f"Multi-head output shape: {multi_head_output.shape}")
print(f"Final output shape: {output.shape}")