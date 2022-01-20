import numpy as np
import math



class neural_network:

    def __init__(self, n_input, n_output):
        self.input_size = n_input
        self.output_size = n_output
        self.layers = []
        self.current_last_n = self.input_size
        self.layer_outputs = []
        np.random.seed(1)

    def add_layer(self, n_neurons, activation="sigmoid"):
        weights = np.random.uniform(low=-0.3, high=0.3, size=(self.current_last_n+1, n_neurons))
        self.current_last_n = n_neurons

        hidden_layer = dict()
        hidden_layer["weights"] = weights
        hidden_layer["activation"] = activation
        self.layers.append(hidden_layer)

    def compile(self):
        self.add_layer(self.output_size, activation="sigmoid")

    def predict(self, input):
        input = np.array(input, dtype=float)
        r, c = input.shape
        input = np.c_[input, np.ones(r)]

        for layer in self.layers:
            output = np.dot(input, layer["weights"])
            output = self.activation(output, layer["activation"])

            self.layer_outputs.append(output)

            if layer != self.layers[-1]:
                r, c = output.shape
                output = np.c_[output, np.ones(r)]

            input = output

        return output

    def train(self, input, labels, epochs, learning_rate):
        input = np.array(input, dtype=float)
        labels = np.array(labels, dtype=float)

        for epoch in range(epochs+1):
            output = self.predict(input)
            delta = []

            errors = np.array(self.layer_outputs[-1] - labels)
            derivatives = self.derivative(self.layer_outputs[-1], self.layers[-1]["activation"])
            delta.append(errors * derivatives)

            for i in range(1, len(self.layer_outputs)):
                errors = np.dot(delta[-1], self.layers[-i]["weights"].T[:,:-1])
                derivatives = self.derivative(self.layer_outputs[-i-1], self.layers[-i]["activation"])
                delta.append(errors * derivatives)

            self.update_weights(input, delta, learning_rate)

            if epoch % 200 is 0:
                #print(output[:3])
                #print(labels[:3])
                #print(np.sum(output[0]))
                sum_error = np.sum((labels - output) ** 2)
                print(f"epoch: {epoch}, error: {sum_error}")



    def update_weights(self, input, delta, learning_rate):

        # add column of ones to input for bias neurons
        i = -1
        r, c = input.shape
        input = np.c_[input, np.ones(r)]

        for layer in self.layers:
            delta_values = delta.pop()

            dot_product = np.dot(input.T, delta_values)
            #print(delta_values.shape)
            #print(dot_product.shape)
            #print(layer["weights"].shape)
            layer["weights"] -= learning_rate * dot_product

            #if layer != self.layers[-1]:
            #layer["weights"][-1,:] -= learning_rate * delta_values[:, :-1].T
            i += 1
            input = self.layer_outputs[i]

            # add column of ones to input for bias neurons
            r, c = input.shape
            input = np.c_[input, np.ones(r)]

        self.layer_outputs = []

    def evaluate(self, inputs, labels):
        outputs = self.predict(inputs)
        outputs = (outputs == outputs.max(axis=1)[:,None])
        correct = np.sum((labels == outputs).all(1))
        total = len(inputs)

        print("\nCorrectly classified: ", correct)
        print("Total eval data: ", total)
        print("Accuracy: ", correct/total)



    def activation(self, inputs, activation):
        if activation is "sigmoid":
            sigmoid_v = np.vectorize(self.sigmoid)
            return sigmoid_v(inputs)
        elif activation is "relu":
            relu_v = np.vectorize(self.relu)
            return relu_v(inputs)
        elif activation is "softmax":
            return self.softmax(inputs)

    def derivative(self, inputs, activation):
        if activation is "sigmoid":
            sigmoid_derivative_v = np.vectorize(self.sigmoid_derivative)
            return sigmoid_derivative_v(inputs)
        elif activation is "relu":
            relu_derivative_v = np.vectorize(self.relu_derivative)
            return relu_derivative_v(inputs)
        elif activation is "softmax":
            return self.softmax_derivative(inputs)

    def error(self, actual, expected):
        return

    def sigmoid(self, x):
        x = 10 if x > 10 else x
        x = -10 if x < -10 else x
        sig = 1 / (1 + math.exp(-x))
        return sig

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def relu(self, x):
        return max(0.0, x)

    def relu_derivative(self, x):
        return (x > 0) * 1

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def softmax_derivative(self, x):
        I = np.eye(x.shape[1], x.shape[0])

        return self.softmax(x) * (I - self.softmax(x).T).T





