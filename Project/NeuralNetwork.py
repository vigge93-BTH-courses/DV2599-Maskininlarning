import numpy as np
from typing import Any, Callable
from enum import auto, Enum
from dataclasses import dataclass


class Activation(Enum):
    SIGMOID = auto(),
    TANH = auto()


class NeuralNetwork():
    @dataclass
    class ActivationFunction():
        func: Callable[[float], float]
        dfunc: Callable[[float], float]

    @staticmethod
    def sigmoid(x):
        np.clip(x, -500, 500)
        return 1/(1+np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x) -> float:
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - x**2

    activationFunctions = {
        Activation.SIGMOID: ActivationFunction(sigmoid.__get__(object), dsigmoid.__get__(object)),
        Activation.TANH: ActivationFunction(
            tanh.__get__(object), dtanh.__get__(object))
    }

    def __init__(self, input_l: int, hidden_l: "list[int]", output_l: int, activation: Activation, lr: float = 0.01):
        self._activationFunction = NeuralNetwork.activationFunctions[activation]
        self._lr = lr

        # Create the layers of the network and initialize them to 0
        self._input_l = np.zeros((input_l, 1))
        self._hidden_l: "list[np.ndarray[np.float64, Any]]" = []
        for h_l in hidden_l:
            self._hidden_l.append(np.zeros((h_l, 1)))
        self._output_l = np.zeros((output_l, 1))

        # Create the weights of the network and initialize them randomly following a uniform distribution
        self._weightsHH: "list[np.ndarray[Any, np.dtype[np.float64]]]" = []
        self._weightsBH: "list[np.ndarray[Any, np.dtype[np.float64]]]" = []
        self._weightsIH = np.random.uniform(-1, 1, (hidden_l[0], input_l))
        for h_l in range(len(hidden_l)-1):
            shape = (hidden_l[h_l+1], hidden_l[h_l])
            self._weightsHH.append(np.random.uniform(-1, 1, shape))
            self._weightsBH.append((np.random.uniform(-1, 1, (shape[0], 1))))
        self._weightsHO = np.random.uniform(-1, 1, (output_l, hidden_l[-1]))
        self._weightBI = np.random.uniform(-1, 1, (hidden_l[0], 1))
        self._weightBO = np.random.uniform(-1, 1, self._output_l.shape)

    def predict(self, x: "list[float]"):
        activation = self._activationFunction.func
        self._input_l = np.array(x).reshape((len(x), 1))

        # Input to hidden
        tempIH = np.matmul(self._weightsIH, self._input_l)
        tempIH = np.add(tempIH, self._weightBI)
        self._hidden_l[0] = activation(tempIH)

        # Hidden to hidden
        for h_l in range(len(self._hidden_l)-1):
            self._hidden_l[h_l +
                           1] = np.matmul(self._weightsHH[h_l], self._hidden_l[h_l])
            self._hidden_l[h_l +
                           1] = np.add(self._hidden_l[h_l+1], self._weightsBH[h_l])
            self._hidden_l[h_l+1] = activation(self._hidden_l[h_l+1])

        # Hidden to output
        self._output_l = np.matmul(self._weightsHO, self._hidden_l[-1])
        self._output_l = np.add(self._output_l, self._weightBO)
        self._output_l = activation(self._output_l)
        return self._output_l.reshape(1, -1).tolist()[0]

    def train(self, x: "list[float]", answers_lst: "list[float]"):
        derivation = self._activationFunction.dfunc
        predictions_lst = self.predict(x)
        predictions = np.array(predictions_lst).reshape(
            (len(predictions_lst), 1))
        answers = np.array(answers_lst).reshape((len(answers_lst), 1))

        # Backpropagate errors
        # Output to hidden
        errorO = answers - predictions
        errorHLast = np.matmul(self._weightsHO.T, errorO)

        # Hidden to Hidden
        errorsH = [errorHLast]
        for n in range(len(self._hidden_l) - 2, -1, -1):
            errorHH = np.matmul(self._weightsHH[n].T, errorsH[0])
            errorsH.insert(0, errorHH)

        # Apply gradient descent
        # Output to hidden
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
    import matplotlib.pyplot as plt
    x = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
    y = [[-1.], [1.], [1.], [-1.]]
    nn = NeuralNetwork(2, [3], 1, Activation.TANH, 0.01)
    for _ in range(100):
        for j in range(100):
            i = j % 4
            nn.train(x[i], y[i])
        pred = []
        for i in range(-50, 50):
            t = []
            for j in range(-50, 50):
                t.append(nn.predict([i/50, j/50])[0])
            pred.append(t)
        pred = (np.array(pred) + 1)/2
        plt.imshow(pred, cmap="winter")
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([0, 100], [0, 1])
        plt.yticks([0, 100], [0, 1])
        plt.draw()
        plt.pause(0.001)
        plt.clf()
        print(f"{nn.predict([-1., -1.])[0]:.3f}", f"{nn.predict([-1., 1.])[0]:.3f}",
              f"{nn.predict([1., -1.])[0]:.3f}", f"{nn.predict([1., 1.])[0]:.3f}")
    pred = []
    for i in range(-50, 50):
        t = []
        for j in range(-50, 50):
            t.append(nn.predict([i/50, j/50])[0])
        pred.append(t)
    plt.imshow(np.array(pred), cmap="winter")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xticks([0, 100], [0, 1])
    plt.yticks([0, 100], [0, 1])
    plt.show()
