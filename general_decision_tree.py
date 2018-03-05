import sys
import numpy as np

class node(object):
	def __init__(self, elements):
		self.elements = elements
		self.category = -1

	def determine_test(self, train_data):
		left, right = [], []
		min_missclass_impurity = 1.0;
		min_test = 0;
		testval = 0;
		rightval = []
		leftval = []
		chkupdates = 0
		for testid in range(9):
			medval = np.median(train_data[self.elements, testid])
			left, right = [], []
			for val in self.elements:
				if train_data[val][testid] < medval:
					left.append(val)
				else:
					right.append(val)

			n1right = np.count_nonzero(train_data[right, 9])
			n1left = np.count_nonzero(train_data[left, 9])

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

	def determine_stop(self, train_data):
		if(len(self.elements) <= 3):
			n1 = np.count_nonzero(train_data[self.elements, 9])
			self.category = 1 if n1 > len(self.elements)-n1 else 0
			return 1

		n1 = np.count_nonzero(train_data[self.elements, 9])
		if(n1 == 0):
			self.category = 0
			return 1
		elif(n1 == len(self.elements)):
			self.category = 1
			return 1
		else:
			return 0


	def train(self, train_data):
		if(self.determine_stop(train_data) == 0):
			if(self.determine_test(train_data) == 0):
				n1 = np.count_nonzero(train_data[self.elements, 9])
				self.category = 1 if n1 > len(self.elements)-n1 else 0
			else:
				self.left_child = node(self.leftval)
				self.right_child = node(self.rightval)
				self.left_child.train(train_data)
				self.right_child.train(train_data)

	def run(self, test_data):
		if(self.category == -1):
			if(test_data[self.testid] < self.testval):
				return self.left_child.run(test_data)
			else:
				return self.right_child.run(test_data)
		else:
			return self.category

train = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, usecols = (range(0,6)+range(7,10)+range(6,7)), dtype=(float, float, int, int, int, int, int, np.dtype('S25'), np.dtype('S25'), int))
test = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1, usecols = (range(0,9)), dtype=(float, float, int, int, int, int, int, np.dtype('S25'), np.dtype('S25')))
train_data = np.zeros([train.shape[0], 10])
test_data = np.zeros([test.shape[0], 9])
word_dict1 = {}
word_dict2 = {}

for i in xrange(train.shape[0]):
	if train[i][-3] not in word_dict1:
		word_dict1[train[i][-3]] = int(len(word_dict1))
	if train[i][-2] not in word_dict2:
		word_dict2[train[i][-2]] = int(len(word_dict2))
	train[i][-3] = int(word_dict1[train[i][-3]])
	train[i][-2] = int(word_dict2[train[i][-2]])
	train[i] = np.array(train[i])	
	for j in xrange(len(train[i])):
		train_data[i][j] = train[i][j]

for i in xrange(test.shape[0]):
	test[i][-2] = int(word_dict1[test[i][-2]])
	test[i][-1] = int(word_dict2[test[i][-1]])
	test[i] = np.array(test[i])
	for j in xrange(len(test[i])):
		test_data[i][j] = test[i][j]


# test_data = np.r_[train_data[11000:11250], test_data]
# train_data = train_data[:11000]

# print test_data.shape, train_data.shape

root = node(range(train_data.shape[0]))
root.train(train_data)
# ncorrect = 0
# for i in test_data:
# 	if(int(i[9]) == root.run(i)):
# 		ncorrect += 1

# print ncorrect, test_data.shape[0]

for i in test_data:
	print root.run(i)