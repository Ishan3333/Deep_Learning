 # A Primer on CNN's:

![picture alt](https://cdn-images-1.medium.com/max/800/1*u2FJVJpUtXN0IHSelbI94A.jpeg)

## There are four main operations in the ConvNet shown above:

___STEP-1: Convolution Operation___

![picture alt](https://cdn-images-1.medium.com/max/600/1*MK7oxI3RZ4_nlMm6bZTQ4A.gif)

___STEP-2: Adding Non-Linearity___

![picture alt](https://cdn-images-1.medium.com/max/800/1*mx9H4SupAed4coT0oaz_5A.png)

___STEP-3: Pooling Operation___

![picture alt](https://cdn-images-1.medium.com/max/800/1*8hgpGxHYcX22zQuvTLLOZQ.png)

___STEP-4: Classification___

![picture alt](https://cdn-images-1.medium.com/max/800/1*ocZWgUk2pPHiB6FB2q_5WA.png)

## Notes

1. All the examples are performed with various datasets:

2. The examples here use Tensorflow as backend for the Keras API

3. Dataset for the example of dog vs cat can be found @ https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/14_page_p8s40_file_1.zip

4. You can tinker around with the things like:

    1. number of convolutional layers

    2. number of filters used in convolutional layer

    3. number of fully connected dense layers

    4. number of nodes in fully connected dense layers

    5. number of pooling layers

    6. number of epochs
  
    7. number of batch size

    8. changing the optimizers

    9. changing the pooling function from "MaxPooling2D" to "MinPooling2D", "AveragePooling2D" etc.

    10. combination of activation functions of different layers

    11. callback monitoring

## References:

    1. https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python#preprocess

    2. https://elitedatascience.com/keras-tutorial-deep-learning-in-python

    3. https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/

    4. https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

    5. https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/

    6. https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

    7. https://medium.com/@siakon/convolutional-nn-with-keras-tensorflow-on-cifar-10-dataset-image-classification-d3aad44691bd

    8. https://parneetk.github.io/blog/cnn-cifar10/

    9. https://github.com/juanlao7/CIFAR100-CNN/blob/master/baseline.py
