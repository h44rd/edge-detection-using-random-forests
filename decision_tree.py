import numpy as np
import random
import pca

class node(object):
	def __init__(self, test_fields, dataset_elements = None, depth = 0):
		self.elements = dataset_elements # List of indices of elements in the current node
		self.category = -1
		self.max_depth = 64
		self.depth = depth
		self.min_elements = 8
		self.z_selection = random.sample([(i, j) for i in range(256) for j in range(256) if i != j], 256)
		self.test_fields = test_fields
		self.z_mapping = np.asarray([])
		self.c_mapping = np.asarray([])
		self.isleaf = 0

	def compute_z(self, label):
		z = []
		for i in self.z_selection:
			if label[i[0]/16][i[0]%16] == label[i[1]/16][i[1]%16] :
				z.append(1)
			else:
				z.append(0)
		return np.asarray(z)

	def get_mapping(self, labels):
		self.z_mapping = []
		self.c_mapping = []
		for i in self.elements:
			temp =  self.compute_z(labels[i])
			self.z_mapping.append(self.compute_z(labels[i]))
		self.z_mapping = np.asarray(self.z_mapping)
		Y, P, mu = pca.pca(np.asarray(self.z_mapping), 1)
		for y in Y:
			self.c_mapping.append(1 if y > 0 else 0)

		self.c_mapping = np.asarray(self.c_mapping)
		#computing pca to log(k) dimensions to compute c

	def determine_test(self, train_data, labels):
		train_data = np.asarray(train_data)
		self.get_mapping(labels)
		left, right = [], []
		min_missclass_impurity = 1.0;
		min_test = 0;
		testval = 0;
		rightval = []
		leftval = []
		chkupdates = 0
		for testid in self.test_fields:
			medval = np.median(train_data[self.elements, testid])
			left_ind, right_ind = [], []
			left, right = [], []
			for ind, val in enumerate(self.elements):
				if train_data[val][testid] < medval:
					left_ind.append(ind)
					left.append(val)
				else:
					right_ind.append(ind)
					right.append(val)
			n1right = np.count_nonzero(self.c_mapping[right_ind])
			n1left = np.count_nonzero(self.c_mapping[left_ind])

			if(len(right) == 0 or len(left) == 0):				
				continue

			Pl = (1.0*n1left)/len(left)

			Pr = (1.0*n1right)/len(right)

			missclass_impurity = 1-max(Pr, Pl)
			if(missclass_impurity < min_missclass_impurity):
				chkupdates = 1
				min_missclass_impurity = missclass_impurity
				min_test = testid
				testval = medval
				rightval = right
				leftval = left

		self.testid = min_test
		self.testval = testval
		self.rightval = rightval
		self.leftval = leftval
		return chkupdates

	def determine_stop(self, train_data, labels):
		if len(self.elements) <= self.min_elements:
			self.category = labels[self.elements[0]]
			self.isleaf = 1
			return 1
		elif self.depth >= self.max_depth:
			self.category = labels[self.elements[0]]
			self.isleaf = 1
			return 1
		else:
			return 0

	def train(self, train_data, labels):
		if self.elements == None:
			self.elements = range(len(train_data))
		if(self.determine_stop(train_data, labels) == 0):
			if(self.determine_test(train_data, labels) == 0):
				self.category = labels[self.elements[0]]
				self.isleaf = 1
			else:
				self.left_child = node(self.test_fields, self.leftval)
				self.right_child = node(self.test_fields, self.rightval)
				self.left_child.train(train_data, labels)
				self.right_child.train(train_data, labels)

	def run(self, test_data):
		if(self.isleaf == 0):
			if(test_data[self.testid] < self.testval):
				return self.left_child.run(test_data)
			else:
				return self.right_child.run(test_data)
		else:
			return self.category