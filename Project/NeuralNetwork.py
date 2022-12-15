import numpy as np
from typing import Any, Callable
from enum import auto, Enum
from dataclasses import dataclass
import math
import random
import matplotlib.pyplot as plt

class Activation(Enum):
    SIGMOID = auto(),
    RELU = auto(),
    # SOFTMAX = auto(), // for later
    TANH = auto()

@dataclass
class ActivationFunction():
    func: Callable[[float], float]
    dfunc: Callable[[float], float]

class NeuralNetwork():
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1/(1+math.exp(-x))
    
    @staticmethod
    def dsigmoid(x: float) -> float:
        return x * (1 - x)
    
    @staticmethod
    def relu(x: float) -> float:
        return max(0, x)
    
    @staticmethod
    def drelu(x: float) -> float:
        return 1.0 if x > 0.0 else 0.0
    
    # @staticmethod // for later
    # def softmax(x: float):
    #     return 

    @staticmethod
    def tanh(x: float) -> float: 
        return np.tanh(x)

    @staticmethod
    def dtanh(x: float):
        return 1 - x**2

    activationFunctions = {
        Activation.SIGMOID: ActivationFunction(sigmoid.__get__(object), dsigmoid.__get__(object)),
        Activation.RELU: ActivationFunction(relu.__get__(object), drelu.__get__(object)),
        Activation.TANH: ActivationFunction(tanh.__get__(object), dtanh.__get__(object))
    }

    def __init__(self, input_l: int, hidden_l: "list[int]", output_l: int, activation: Activation, lr: float = 0.01):
        self._activationFunction = NeuralNetwork.activationFunctions[activation]
        self._lr = lr

        self._input_l = np.zeros((input_l, 1))# We initilize the input as a as an input layer of 0s
        self._hidden_l: "list[np.ndarray[np.float64, Any]]" = [] # The hidden layer is a list of arrays that contain the amount of nodes
        for h_l in hidden_l: #Here we create a matrix for each hidden layer in the list, array -> matrix 
            self._hidden_l.append(np.zeros((h_l, 1)))
        self._output_l = np.zeros((output_l, 1)) #We do the same as in the input layer
        self._weightsHH: "list[np.ndarray[Any, np.dtype[np.float64]]]" = []
        self._weightsBH: "list[np.ndarray[Any, np.dtype[np.float64]]]" = []
        self._weightsIH = np.random.uniform(-1, 1, (hidden_l[0], input_l)) #We create a randomized matrix from the normal dist to create a matrix with hidden layer rows and input layer columns, aka 3x3
        for h_l in range(len(hidden_l)-1):
            shape = (hidden_l[h_l+1], hidden_l[h_l])
            self._weightsHH.append(np.random.uniform(-1, 1, shape))
            self._weightsBH.append((np.random.uniform(-1, 1, (shape[0], 1))))
        self._weightsHO = np.random.uniform(-1, 1, (output_l, hidden_l[-1]))
        
        self._weightBI = np.random.uniform(-1, 1, (hidden_l[0], 1))
        self._weightBO = np.random.uniform(-1, 1, self._output_l.shape)


    def predict(self, x: "list[float]"):
        activation = np.vectorize(self._activationFunction.func)
        inputs = np.array(x).reshape((len(x),1))
        assert inputs.shape[0] == self._input_l.shape[0]
        tempIH = np.matmul(self._weightsIH, inputs) #we muliply the weights of the input -> hidden with inputs to give us a matrix of 3x2
        tempIH = np.add(tempIH, self._weightBI)

        self._hidden_l[0] = activation(tempIH)
        for h_l in range(len(self._hidden_l)-1):
            self._hidden_l[h_l+1] = np.matmul(self._weightsHH[h_l], self._hidden_l[h_l]) #We multiply the weights of the hidden layer by the hidden layer 
            self._hidden_l[h_l+1] = np.add(self._hidden_l[h_l+1], self._weightsBH[h_l]) #We add the weights of the hidden layer by the hidden layer 
            self._hidden_l[h_l+1] = activation(self._hidden_l[h_l+1])
        
        self._output_l = np.matmul(self._weightsHO, self._hidden_l[-1])
        self._output_l = np.add(self._output_l, self._weightBO)
        self._output_l = activation(self._output_l)
        return self._output_l.reshape(1, -1).tolist()[0]

    def train(self, x: "list[float]", answers_lst: "list[float]"):
        derivation = np.vectorize(self._activationFunction.dfunc)
        predictions_lst = self.predict(x)
        predictions = np.array(predictions_lst).reshape((len(predictions_lst),1))
        answers = np.array(answers_lst).reshape((len(answers_lst),1))
        #Backpropagate errors
        errorO = answers - predictions
        errorHLast = np.matmul(self._weightsHO.T, errorO)

        errorsH = [errorHLast]
        for n in range(len(self._hidden_l) - 2, -1, -1):
            errorHH = np.matmul(self._weightsHH[n].T, errorsH[0])
            errorsH.insert(0, errorHH)
        
        # adjust weights
        gradient = derivation(self._output_l)
        gradient *= errorO * self._lr

        self._weightBO += gradient
        deltaHidden = np.matmul(gradient, self._hidden_l[-1].T)
        self._weightsHO += deltaHidden
        
        # Hidden to hidden
        for n in range(len(self._hidden_l)-1, 0, -1):
            gradient = derivation(self._hidden_l[n])
            gradient *= errorsH[n]*self._lr

            self._weightsBH[n-1] += gradient
            self._weightsHH[n-1] += np.matmul(gradient, self._hidden_l[n-1].T)

        # Hidden to input
        gradient = derivation(self._hidden_l[0])
        
        gradient *= errorsH[0] * self._lr
        self._weightBI += gradient
        self._weightsIH += np.matmul(gradient, self._input_l.T)


if __name__ == '__main__':
    x = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    y = [[1], [1], [1], [-1]]
    nn = NeuralNetwork(2, [6], 1, Activation.TANH, 0.01)
    # print(nn.predict([0., 0.]), nn.predict([0., 1.]), nn.predict([1., 0.]), nn.predict([1., 1.]))
    while True:
        for _ in range(5000):
            i = random.randint(0, 3)
            nn.train(x[i], y[i])
        pred = []
        for i in range(20):
            t = []
            for j in range(20):
                t.append(nn.predict([i/20, j/20])[0])
            pred.append(t)
        plt.imshow(np.array(pred), cmap="winter")
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        print(f"{nn.predict([0., 0.])[0]:.3f}", f"{nn.predict([0., 1.])[0]:.3f}", f"{nn.predict([1., 0.])[0]:.3f}", f"{nn.predict([1., 1.])[0]:.3f}")