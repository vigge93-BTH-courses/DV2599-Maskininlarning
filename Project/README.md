# Required libraries
## NeuralNetwork.py
To use the nerual network class, you need the following libraries:
* Numpy

To run the XOR example in the file, you need the following libraries:
* Numpy
* Matplotlib

## MNIST.ipynb
To run the notenook with the MNIST example, you need the following libraries:
* Jupyter Notebook
* Numpy
* Pandas
* Matplotlib
* Scikit-learn

# Using the neural network class
To use the nerual network class, you need to instansiate a NeuralNetwork object. The constructor takes four parameters required parameters and one optional parameter. These are the following:
* input_l: int - The number of nodes in the input layer
* hidden_l: list[int] - A list of integers specifying the number of nodes in each hidden layer
* output_l: int - The number of nodes in the output layer
* activation: Activation: The activation function to use. Use the Activation enum in the same module.
* lr: Optional[float] - The learning rate for the model

The class provides two functions, one for making predictions and one for training the model.

The prediction function has one parameter, a list of the values for the data point that you want to predict. The function returns a list of values representing each of the nodes in the output layer.

The training function takes two parameters, a list of the values for the data point that you want to train, and a list of the expected outputs of the network. The function does not return anything.

# Using the notebook
Open the notebook and run it. With the current parameters, the training will take about 60 minutes to complete.

A pickled file is provided with the best model in the report, and can be loaded for testing in the notebook without having to retrain the model.
