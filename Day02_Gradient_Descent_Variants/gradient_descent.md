## Gradient descent variants
- sgd (normal gradient descent) has some problems and performs bad in high dimensional landscapes
- The problems are : 
1. step size : fixed step size no matter the scenario
2. direction : gradient tells where the slope is steep not the actual direction we need to go so instead of moving forward we oscillate left and right. (Ravine problem)
3. memory : no memory between steps

These are solved by using momentum and adam.

# Momentum
we keep running average of gradients called velocity.
velocity = beta * velocity + (1 - beta) * gradient  ; beta generally 0.9
weight   = weight - lr * velocity

# Adam
it allows adaptive learning rate for each weights
m = running average of gradients         (like momentum — direction)
v = running average of gradient squared  (how large gradients have been)
m = beta1 * m + (1 - beta1) * gradient        # momentum term
v = beta2 * v + (1 - beta2) * gradient²       # squared gradient term

m_corrected = m / (1 - beta1^t)               # bias correction
v_corrected = v / (1 - beta2^t)               # bias correction

weight = weight - lr * m_corrected / (sqrt(v_corrected) + epsilon)
beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8


# Insights 
- For this simple problem sgd performed well than both momentum and adam. Adam's performance was worst for this simple problem.
- Adam is built for noisy, large scale problems with sparsity so we get its advantage on such data.