import numpy  as np
import pandas as pd
import glob

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

import warnings
warnings.filterwarnings('ignore')


from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

from scipy.signal import convolve2d
from scipy.stats import gaussian_kde as kde



class Credit(object):
	"""docstring for Credit"""
	def __init__(self, root):
		self.root = root
		self.train_images = glob.glob(self.root+'/test_images/*')

		for i in self.train_images:
			image = Image.imread(i)
			image = self.normalize(image)
			# image = self.conv2d(image)

			# image = 255 * (image > 150)

			plt.imshow(image, cmap='gray')
			plt.show()


	def normalize(self, image):
		gray = np.array([0.299, 0.587, 0.114])
		image = np.dot(image, gray)
		image = image**2

		image = ((image - image.min())/(image.max() - image.min())*255).round()
		return image

	def conv2d(self, image):
		sobel_x = np.c_[
			[1, 1, 1],
			[1, -8, 1],
			[1, 1, 1]
		]

		sobel_y = np.c_[
			[-1, 0, 1],
			[-2, 0, 2],
			[-1, 0, 1]
		]
		image = convolve2d(image, sobel_x, mode='same', boundary='symm')

		return image

	def Laplace_suanzi(self, img):
		r, c = img.shape
		new_image = np.zeros((r, c))
		L_sunnzi = np.array([
			[0,-1,0],
			[-1,4,-1],
			[0,-1,0]
		])     
		L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])      
		for i in range(r-2):
			for j in range(c-2):
				new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
		return np.uint8(new_image)
		

dataset_path = '/home/jonty/Downloads/资料下载'

if __name__ == '__main__':
	test = Credit(dataset_path)