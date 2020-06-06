


> Written with [StackEdit](https://stackedit.io/).
# Fashon-MNIST project
## Introduction
This project's first objective is to test one of previously implemented models using Fashion-MNIST dataset. I decided to test k-NN algoritm (k-nearest neighbors algorithm).
The second objective is to create my own implementation of model which would categorise clothes from Fashion-MNIST.

## Methods
### k-NN
The only problem with previous implementation of k-NN is that it uses **hamming distance metric** which is used to compare two **binary** pieces of information. Fashion-MNIST dataset contains grayscale images which, as you can see [here](https://github.com/zalandoresearch/fashion-mnist), are not discrete. In order to use algorithm witch this metric you have to preprocess images doing **binarization**. Obviously, the other solution is changing the metric. For example you can use **euclidean distance metric**.

In my research I'll try those two approaches to see, which will produce better results. Of course first approach depends on threshold used within binarization algorithm (simply if pixel value is greather then threshold it will become 1 else 0) so I'll try to find which is best for Fashion-MNIST dataset.

You can see source kod of k-NN implementation in **K_NN.py** file and tests in **K_NN_Tests.py**.
### [SGD in tensorflow implementation of keras] zmienić
My model is sequential neural network model implemented using keras. 'Sequential' means, that neurons belong to layers - input layer, output layer and hidden layers between them. You could think of this model type as 'Hello World' of neural network machine learning. Each neuron is connected to all neurons from the previous layer and each connection has a weight. Basically, process of 'teaching' a model is a manipulation of those weights and biases used in their's activation functions.

While creating even simple sequential model you have to consider what function or functions you'll use as activation functions and how you want to initialize biases and weights. There are several function used commonly in deep learning as activation fuctions. Those are:

 - sigmoid function
$$\ \sigma(x) =\frac{1}{1+e^{-x}}$$
 - ReLu - 
$$\ f(x) = max(0,x)$$
 - Leaky Relu - 
$$\ f(x) =max(0.01x,x)$$
 - Tanh(x) 
$$\ Tanh(x)$$

You can learn more about them by watching [Stanford university lectures](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6).
All of them have pros and cons, but as a rule of thumb you should never use sigmoid function and use ReLu instead. It's also possible to experiment with usage of other functions.

Those lectures also contain basic ideas how to initialize weights. The common solution is Xavier initialization (also known as Glorot initialization), however it woked well with sigmoid function not a ReLu function. You should use slightly modified version called he et al initialization.

#### about model
#### Gabor filter
Gabor filter is a linear filter used for preprocessing images for deep learning. To visualise it's effects you can look at images below.

![](/data/images/gabor_filter_1.png)
![](/data/images/gabor_filter_2.png)
![](/data/images/gabor_filter_3.png)

As you can see, gabor filter extracts features from image, however you have to know in which direction you want them extracted. **Fashon-MNIST contains only non rotated images**, so you can apply horizontal filter.

Below you can see Fashon-MNIST images before and after applying gabor filter.

![](/data/images/gabored_fashon_mnist.png)

#### Extending training set
It is possible to increase size of training set by applying small changes to images that won't change their classification. It's a common technique that should increase accuracy of model (if done correctly). As I mentioned before, Fashon-MNIST contains only non rotated images, so applying rotation is a bad idea. However you can still apply translation by vector.

Below you can see Fashon-MNIST images before and after applying translation by random vectors containing values from -3 to 3.

![](/data/images/translated_fashon_mnist.png)