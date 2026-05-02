backprop is used to propagate the gradient of error backward from output layer to earlier layer so the weights get adjusted reducing the error.
The gradients shape needs to match the shape of the variable they are updating so there is use of transpose in matrix form.
shape for weights = (no.of inputs, no.of neurons)