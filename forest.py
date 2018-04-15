import numpy as np
import decision_tree
import preprocess_data
import random

class forest(object):
	def __init__(self, T = 4):
		# We randomly select sqrt(N) features to take as possible tests for each tree
		self.trees = [decision_tree.node(random.sample(range(7228), 85)) for i in range(2*T)] 

	def train(self, train_data):
		train_data_x, train_data_y = preprocess_data.get_train_data(train_data)
		for i in self.trees:
			i.train(train_data_x, train_data_y)

	def run(self, img):
		img = preprocess_data.rgb2luv(img)
		img = preprocess_data.test_preprocessing1(img)
		img = np.asarray(img)
		output = np.zeros((int(len(img)/2), int(len(img[0])/2)))
		averaging = np.ones((int(len(img)/2), int(len(img[0])/2)))
		trees_used = [0] # Alternating between 4 trees every step
		for x in range(0, len(img)-32, 8):
			for y in range(0, len(img[0])-32, 8):
				for ind in trees_used:
					tree = self.trees[ind]
					features = preprocess_data.test_preprocessing2(img[x:x+32, y:y+32, :])
					output[x/2:(x+32)/2, y/2:(y+32)/2] = np.add(output[x/2:(x+32)/2, y/2:(y+32)/2], np.asarray(tree.run(features)))
					averaging[x/2:(x+32)/2, y/2:(y+32)/2] += 1
					trees_used = [(i+1)%4 for i in trees_used]
		output = np.divide(output, averaging)
		return averaging


