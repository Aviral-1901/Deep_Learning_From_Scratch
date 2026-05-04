# Activation Functions
Activations functions are used to add non-linearity. Without activation functions the whole network collapses into into one single linear transformation

## Sigmoid function
sigmoid(z) = 1 / (1 + exp^(-z))                                             
d_sigmoid = sigmoid(z) * (1 - sigmoid(z))                                                  
It keeps any input to fit between 0 and 1. At very big or very small values its derivative becomes 0 and zero gradient means no learning.
Sigmoid output is always positive so all the gradients for that layers weights are of same sign which causes either everything to go up or all to go down together. We cannot update some weights up and some down independently.


## Tanh
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))                                                  
d_tanh = 1 - tanh(z)**2                                                                
output range is (-1, 1)
outputs are now -ve and +ve so weights can either go up or down independently
derivative of tanh at extremes is still zero.


## ReLU
ReLU(z) = max(0, z)                                                                       
d_relu = 1 if z > 0 else 0                                                                          
solves vanishing gradient problem                        
if neuron receives -ve input then it outputs 0 and its gradient is also 0 so weight do not update and if thi keeps happening, the neuron stops contributing (dying neurons)             


## GELU
GELU(z) = z × sigmoid(1.702 × z)                       
d_gelu = (gelu(z + 0.001) - gelu(z)) / 0.001            
gelu is smooth version of relu.                        
neurons in negative region dont become fully 0 and receive small gradient signal so the neuron is not effectively dead. Therefore transformers use it.                                            


