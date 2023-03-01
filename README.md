# VGG19
This is a Python script that trains a VGG19 model on image data using Keras. The code is divided into several parts, each of which performs a specific task.

First, several packages and modules are imported, including optimizers, losses, metrics, and ImageDataGenerator from Keras, as well as the VGG19 model from Keras Applications. The code then mounts a Google Drive to access the training and test data.

Next, an instance of the ImageDataGenerator class is created for the training data, and the training data is loaded using the flow_from_directory method. The same is done for the test data.

The VGG19 model is loaded with pre-trained weights from the ImageNet dataset, and its summary is printed. Then, the layers of the VGG19 model up to the 19th layer are frozen to prevent them from being retrained. A new output layer with two nodes and a softmax activation function is added, and the modified model is compiled with the mean squared error loss function, stochastic gradient descent optimizer, and categorical accuracy metric.

The model is then trained using the fit_generator method with a batch size of 32, two epochs, and the training and test data. The fit_generator method also takes a ModelCheckpoint callback to save the best model weights during training, and an EarlyStopping callback to stop training if the validation accuracy does not improve after 20 epochs. The training history is then plotted for accuracy and loss.

In the second example of the code, the same ImageDataGenerators and VGG19 model are loaded, but this time the model is compiled and trained with the fit_generator method without any callbacks. The training history is plotted again for accuracy and loss.

In the final part of the code, a TensorFlow image data generator is created for the training data with various data augmentation parameters such as rotation, width and height shifts, and horizontal flipping. Then, the training, validation, and test sets are loaded using their respective data generators and rescaled to a range of 0 to 1.


The first block of code imports necessary libraries for building and training a deep learning model using Keras API of Tensorflow library. These libraries include:

optimizers: a sub-library of tensorflow.keras containing various optimization algorithms.
losses, metrics, Model, and Dense: sub-libraries of keras for defining loss functions, evaluation metrics, and layers of a neural network.
ImageDataGenerator: a class for generating image data in batches with real-time data augmentation.
image: a module from keras.preprocessing for performing image-related tasks.
python

The next block of code creates an instance of ImageDataGenerator with no image augmentation, trdata, for the training set and another instance with the same characteristics, tsdata, for the test set. Then it uses the flow_from_directory method of these instances to create iterators for training and test sets that will yield image data in batches. The images in the specified directories are resized to 224x224 pixels during this process.


The next block of code loads a pre-trained VGG19 model with weights trained on ImageNet dataset, including the fully connected top layer that contains 1000 neurons for classifying ImageNet images into 1000 classes. The code then prints out the summary of the model.


The next block of code sets the first 19 layers of the VGG19 model as non-trainable since they have already been trained on ImageNet and contain important features. The last fully connected layer of the model is removed, and a new dense layer with two neurons and softmax activation function is added to classify our two target classes.


The next block of code compiles the model with mean squared error loss function, stochastic gradient descent optimizer, and categorical accuracy metric.


The next block of code trains the model for 2 epochs on the training set and uses the test set for validation. It also saves the model weights to a file named "vgg19New.h5" whenever the validation categorical accuracy improves.




The model is trained on a training dataset and evaluated on a separate test dataset using mean squared error as the loss function and categorical accuracy as the evaluation metric. The code includes data augmentation techniques to increase the size of the training dataset, and includes callbacks to save the best performing model and to stop the training process early if performance on the validation dataset plateaus. The code also includes code to plot the training and validation accuracy and loss curves, as well as code to load images from a directory and preprocess them for input into the model.
