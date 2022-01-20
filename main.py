import neural_network as nn
import numpy as np
import csv
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    inputs = []
    labels = []

    with open('mnist.csv') as csvfile:
        data = csv.reader(csvfile)
        next(data, None)
        for row in data:
            labels.append(row.pop(0))
            inputs.append(list(map(int, row)))

    input_size = len(inputs[0])
    encoder = OneHotEncoder(sparse=False)
    onehot_encoded_labels = encoder.fit_transform(np.array(labels).reshape(-1,1))
    inputs = np.array(inputs, dtype=float)
    inputs = inputs / 256
    # print(inputs[0])
    onehot_encoded_labels = np.array(onehot_encoded_labels, dtype=float)
    # print(type(onehot_encoded_labels))

    network = nn.neural_network(input_size, 10)
    network.add_layer(8, activation="sigmoid")
    network.add_layer(4, activation="sigmoid")
    network.compile()

    network.train(inputs[:3000], onehot_encoded_labels[:3000], 4000, 0.01)

    network.evaluate(inputs[8000:], onehot_encoded_labels[8000:])