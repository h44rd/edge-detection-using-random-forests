import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import forest
import pickle

def get_train_data(dataset_dir="input"):   
	image_dir = os.path.join(dataset_dir, "images", "train")
	label_dir = os.path.join(dataset_dir, "groundTruth", "train")
	data = []

	for file_name in os.listdir(label_dir):
		gts = io.loadmat(os.path.join(label_dir, file_name))
		gts = gts["groundTruth"].flatten()
		bnds = [gt["Boundaries"][0, 0] for gt in gts]
		segs = [gt["Segmentation"][0, 0] for gt in gts]

		img = imread(os.path.join(image_dir, file_name[:-3] + "jpg"))
		img = img_as_float(img)

		data.append((img, bnds, segs))
		break
	return data

def test_model(model, input_dir="input", output_dir="edges"):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	image_dir = os.path.join(input_dir, "images", "test")
	file_names = filter(lambda name: name[-3:] == "jpg", os.listdir(image_dir))
	n_image = len(file_names)

	for i, file_name in enumerate(file_names):
		img = img_as_float(imread(os.path.join(image_dir, file_name)))

		edge = np.array(model.run(img))
		edge /= np.max(abs(edge))
		edge = img_as_ubyte(edge)
		# print edge
		if not os.path.exists(os.path.join(output_dir, file_name[:-3] + "png")):
			imsave(os.path.join(output_dir, file_name[:-3] + "png"), edge)


if __name__ == "__main__":
	forest = forest.forest()
	if not os.path.exists("./model/forests/forest.h5"):
		print "Weights not present, training using mini-dataset"
		train_data = get_train_data()
		forest.train(train_data)
		if not os.path.exists(os.path.dirname("./model/forests/")):
			os.makedirs(os.path.dirname("./model/forests/"))
		print "Saving Trained Model"
		with open("./model/forests/forest.h5", "wb") as f:
			pickle.dump(forest, f, pickle.HIGHEST_PROTOCOL)

	else:
		try:
			with open("./model/forests/forest.h5", "rb") as f:
				forest = pickle.load(f)
		except:
			train_data = get_train_data()
			forest.train(train_data)

	test_model(forest)