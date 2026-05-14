import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model) :
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len) :
        for i in range(0, d_model, 2) :
                PE[pos, i] = np.sin(pos / 10000 ** (i / d_model))

                if i + 1 < d_model :
                     PE[pos, i+1] = np.cos(pos / 10000 ** (i / d_model))
    
    return PE

PE = positional_encoding(50, 8)
X = np.array([
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

PE3 = positional_encoding(3, 8)
X_with_position = X + PE3

print("X original:")
print(X)
print("\nPositional encoding for 3 positions:")
print(PE3)
print("\nX with position added:")
print(X_with_position)
plt.figure(figsize=(10,6))
plt.imshow(PE, aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional encoding')
plt.show()

