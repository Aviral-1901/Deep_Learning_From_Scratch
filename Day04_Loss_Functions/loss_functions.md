# Loss Functions 
They tell how far we are from the actual value basically the error.

## MSE (Mean Squared Error)
MSE loss = (predicted - true)^2
MSE gradient = 2*(predicted - true)
When the model predicts(sigmoid output) around 0 or around 1 the gradients are very small. It is ok for correct value as we are near the true value so it is good that the we dont need big changes but when we are extremely far from true value then at such case the gradient is also very small so instead of changing fast for huge error, the model barely improves.

## Cross-Entropy Loss
L = -[y_true × log(predicted) + (1-y_true) × log(1-predicted)]
When the model is very wrong then we get huge loss and when it is near to true value then we get tiny loss which is what we want.
When we combine cross entropy with sigmoid then the gradient becomes very simple which is           
CE_gradient = predicted - y_true
So when model is very wrong we get large gradient so learning becomes faster and when model is confidently right we get small gradient so there is barely an update.
