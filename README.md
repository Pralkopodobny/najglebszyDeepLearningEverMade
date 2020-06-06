


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
### My model
My model is sequential neural network model implemented using keras. 'Sequential' means, that neurons belong to layers - input layer, output layer and hidden layers between them. You could think of this model type as 'Hello World' of neural network machine learning. Each neuron is connected to all neurons from the previous layer and each connection has a weight. Basically, process of 'teaching' a model is a manipulation of those weights and biases used in their's activation functions.

While creating even simple sequential model you have to consider what function or functions you'll use as activation functions and how you want to initialize biases and weights. There are several function used commonly in deep learning as activation fuctions. Those are:

 - sigmoid function: <img src="https://render.githubusercontent.com/render/math?math=\sigma(x) =\frac{1}{1 %2B e^{-x}}">
 - ReLu: <img src="https://render.githubusercontent.com/render/math?math=f(x) = max(0,x)">
 - ReLu: <img src="https://render.githubusercontent.com/render/math?math=f(x) = max(0.01x,x)">
 - Tanh

You can learn more about them by watching [Stanford university lectures](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6).
All of them have pros and cons, but as a rule of thumb you should never use sigmoid function and use ReLu instead. It's also possible to experiment with usage of other functions.

Those lectures also contain basic ideas how to initialize weights. The common solution is Xavier initialization (also known as Glorot initialization), however it woked well with sigmoid function not a ReLu function. You should use slightly modified version called he et al initialization.

#### about model
My model is based of 
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

## Results

### k-NN with hamming metric
I [binarized](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6) fashon-MNIST images with diffrent thresholds and I used [k-NN](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6) model to classify them. You can see results in table below and sorce code of tests [here](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6).
| Threshold  | k_values | Acuraccy | Best k |
|--|--|--|--  |
| 0.1 | 3-9 | 0.8588 | 3 |
| 0.2 | 3-9 | 0.8528 | 5 |
| 0.3 | 3-9 | 0.8413 | 7 |
| 0.4 | 3-9 | 0.8217 | 5 |
| 0.5 | 3-9 | 0.7958 | 5 |
| 0.6 | 3-9 | 0.7641 | 7 |
| 0.7 | 3-9 | 0.716 | 9 |
| 0.8 | 3-9 | 0.6378 | 9 |
| 0.9 | 3-9 | 0.5244 | 7 |

As we can see the best results are with the smallest threshold.
![](/data/images/knn_winners.png)
### k-NN with euclidean metric
I normalized values of pixels to 0-1 range and I used model to classify them. You can see results in table below and sorce code of tests [here](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6).
| HH:MM:SS | k_values | Best Acuraccy  | best k
|--|--|--|--|
| 00:11:00 | 3-9 | 0.8527 | 3 |

