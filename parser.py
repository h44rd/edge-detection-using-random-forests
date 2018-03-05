import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# D = io.loadmat('new.mat')
# Im = D['images']
#
# implot = plt.imshow(Im[:,:,:,0])

class Dataset:
    def __init__(self,file):
        self.data = io.loadmat(file)
        self.images = self.data['images']
        self.depths = self.data['depths']
        self.labels = self.data['labels']
        # implot = plt.imshow(self.images[:,:,:,0])

    def show_image(self,i):
        implot = plt.imshow(self.images[:,:,:,i])
        plt.show()

    def  show_depth(self,i):
        depths = self.depths[:,:,:,i]/np.linalg.norm(self.depths[:,:,:,i])
        depths = 255.0*depths
        implot = plt.imshow(depths)
        plt.show()

    def show_label(self,i):
        implot = plt.imshow(self.labels[:,:,i])
        plt.show()

D = Dataset('new.mat')
D.show_label(0)
D.show_label(1)
D.show_label(2)
