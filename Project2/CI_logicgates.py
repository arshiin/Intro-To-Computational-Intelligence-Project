import numpy as np
import matplotlib.pyplot as plt


def perceptron(input, weight, learning_rate, expected_output, iterations):
    mean_err = np.zeros((iterations, 3))
    for i in range(iterations):
        output = np.dot(input, weight)
        output = np.heaviside(output, 1)
        err = expected_output - output
        weight = weight + np.dot(learning_rate * input.transpose(), err)
        mean_err[i:i + 1, 0] = abs(err[:, 0]).mean()
        mean_err[i:i + 1, 1] = abs(err[:, 1]).mean()
        mean_err[i:i + 1, 2] = abs(err[:, 2]).mean()
    print(weight)
    return mean_err


x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
d = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0]])
n = 0.001
w = np.random.random((3, 3)) * n
print(w)

train = perceptron(x, w, n, d, 100)


plt.figure()
plt.subplot(3, 1, 1)
plt.title('AND Gate')
plt.plot(train[:, 0])

plt.subplot(3, 1, 2)
plt.title('OR Gate')
plt.plot(train[:, 1])

plt.subplot(3, 1, 3)
plt.title('XOR Gate')
plt.plot(train[:, 2])

plt.tight_layout()
plt.show()
