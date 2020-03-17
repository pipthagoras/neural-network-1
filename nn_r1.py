import numpy as np
import random
import time

class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b.repeat(a.shape[1], axis=1))
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
		t_start = time.time()
		for i in range(0, epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, len(training_data), mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			print "Epoch ", i + 1, " / ", epochs, " Success rate: ", self.evaluate(test_data)
		print time.time() - t_start

	def update_mini_batch(self, mini_batch, eta):
		x = np.concatenate([p[0] for p in mini_batch], axis=1)
		y = np.concatenate([p[1] for p in mini_batch], axis=1)
		nabla_b, nabla_w = self.backprop(x, y)
		self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]



	def cost_derivative(self, output_activations, y):
		return output_activations - y

	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# Feedforward
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b.repeat(activation.shape[1], axis=1)
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# Backward pass
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta.sum(axis=1, keepdims=True)
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.num_layers):
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
			nabla_b[-l] = delta.sum(axis=1, keepdims=True)
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
		return sum(float(x == y) for x, y in test_results) / len(test_data)


def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))