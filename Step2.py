import numpy as np
import matplotlib.pyplot as pyplot
#              Age  Gen Weight Bp down   Bp up  sym Dia
X = np.array(([0.01, 0, 5/120, 80/150, 90/150, 0.1, 0], 
			  [0.02, 1, 4/120, 70/150, 80/150, 0.1, 0], 
			  [0.03, 1, 5/120, 75/150, 80/150, 0.1, 0], 
			  [0.02, 0, 5/120, 50/150, 70/150, 0.1, 0], 
			  [0.03, 1, 4/120, 40/150, 90/150, 0.1, 0], 
			  [0.05, 0, 10/120, 70/150, 90/150, 0.1, 1], 
			  [0.03, 1, 5/120, 90/150, 90/150, 0.1, 1], 
			  [0.05, 1, 40/120, 60/150, 100/150, 0.1, 1], 
			  [0.05, 0, 50/120, 70/150, 100/150, 0.1, 0],
			  [0.01, 1, 3/120, 60/150, 90/150, 0.1, 0],
			  [0.50, 0, 70/120, 90/150, 120/150, 0.6, 1]
			  ), dtype=float)
# 				Drugtype, Dosage
Y = np.array(([0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0.1, 0.1],
			  [0, 0]
	), dtype=float)
class Network:
	def __init__(self):
		self.inputLayerSize = 7 
		self.outputLayerSize = 2
		self.hiddenLayerSize = 5
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
	def forward(self, X):
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat
	def sigmoid(z):
		return 1/(1+np.exp(-z))
	def sigmoidPrime(z):
		return np.exp(-z)/(1+np.exp(-z)**2)
	def costFunctionPrime(self, X, Y)
		self.yHat = self.forward(X)
		delta3 = np.multiply(-(y-self.yHat))
NN = Network()
yHat = NN.forward(X)
