# OverFitting
It is the condition when the network learns the training dataset including the noise and unnecessary data instead of generalizing for the data.
It mainly occurs if data size is small, network is bigger.

## Bias-Variance tradeoff
If there are few parameters network cannot learn anything useful so high bias. If there are too many parameters relative to data then network memorizes everything so high variance.
We want both of them balanced.

## Early stopping
We stop the training after validation loss starts increasing even if the training loss is going down. We dont immediately stop learning, we wait for few epochs as loss can fluctuate then after that we stop the training.
Patience = how long to wait after validation loss starts increasing before we stop the training

## Dropout
During training we randomly turn off some of the neurons so that neurons do not co-adapt and neurons become independent of each other. We use dropout rate which controls how many neurons are being turned off. For this we generate a mask then apply the mask to the neurons during the training.                                                       
mask = np.random.rand(*Out1.shape) > dropout_rate  //Here we generate random numbers of out1 shape and check if they are greater than dropout_rate and we get an array of 0 and 1 like [0,1,1,0,1,0] where the 0 ones are being turned off.


# L2 Regularization
We add a penalty term to the loss so that weights dont become too large. It is used to make prediction error and keep weights small so that weights dont become large to fit the training data perfectly.                                                                                          
L2_penalty =  λ * sum(weights^2)                                                                    

Loss = Loss + L2_penalty                                                                     
weight update is done by :                                                              
W = W - learning_rate * gradient_from_loss - lr * 2 *  λ * W