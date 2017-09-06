# Learning-Deep-Learning
A repository containing tutorials and examples that I have gone through in exploring deep learning

## [MNIST in Tensorflow](https://github.com/brianhhu/Learning-Deep-Learning/blob/master/MNIST/MNIST%20in%20Tensorflow.ipynb)
A simple example of using Tensorflow to classify handwritten digits from the MNIST dataset. We begin with a simple 1-layer fully-connected network, and expand to using deep convolutional neural networks. In the process, we explore different optimization techniques (gradient descent vs. Adam), and also useful tricks such as learning rate decay and dropout. We achieve a final test accuracy greater than 99%.

## [FCN for Segmentation in Keras](https://github.com/brianhhu/Learning-Deep-Learning/blob/master/Segmentation/Fully%20Convolutional%20Neural%20Networks.ipynb)
We use the state-of-the-art ResNet50 network and turn it into a fully convolutional neural network for weak object segmentation. We do this by removing the last average pooling layer and changing the last dense layer into a convolutional layer, which can now be applied at multiple spatial locations across the image. We define a custom soft-max layer to get the probabilities across classes at each spatial location. We test our model on an image of a dog, and combine the segmentation results across three different scales.

## [Fine-Tuning and Transfer Learning in Tensorflow](https://github.com/brianhhu/Learning-Deep-Learning/blob/master/TransferLearning/Oxford-17%20Fine%20Tune.ipynb)
An example of how to use a pre-trained network structure  (AlexNet) with the corresponding weights and simply fine-tune the final fully-connected layer for a new dataset. Here, we used a Caffe model trained on the Oxford-102 flower dataset, convert it into a Tensorflow model, and re-train it on the smaller Oxford-17 flower dataset. We achieve a final mean test accuracy of 91.7% averaged across all three splits of the data. The code associated with this exercise was developed as part of a take-home assignment for an interview.
