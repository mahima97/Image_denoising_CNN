from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.callbacks import Callback
from keras.datasets import mnist
import numpy as np

from keras.callbacks import Callback
# import wandb
from data_load import data_load
import cv2
import os
# from wandb.keras import WandbCallback

batch_size = 8
fname = 'val2017'
# import add_noise
class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		# call the parent constructor
		super(Callback, self).__init__()
		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt
	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath,
				"epoch_{}.h5".format(self.intEpoch + 1)])
			self.model.save(p, overwrite=True)
		# increment the internal epoch counter
		self.intEpoch += 1



# def add_noise(x_train, x_test):
# 	# Add some random noise to an image

# 	noise_factor = 0.5
# 	x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
# 	x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# 	x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# 	x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# 	return x_train_noisy, x_test_noisy

# run = wandb.init()
# config = run.config

# config.encoding_dim = 32
# config.epochs = 10


x_train,x_test = data_load(fname)
print(x_train[0])
# (x_train, _), (x_test, _) = mnist.load_data()
# (x_train_noisy, x_test_noisy) = add_noise(x_train, x_test)

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

model = Sequential()
model.add(Reshape((256,256,3), input_shape=(256,256,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(Reshape((256,256,3)))
model.compile(optimizer='adam', loss='mse')







print(model.summary())

def image_generator(files,batch_size = 8):
	noise_factor = 0.5
	while True:
		# print(files[0])
		num = batch_size
		noise_factor = np.random.rand(num)
		idx = np.arange(0 , len(files))
		np.random.shuffle(idx)
		idx = idx[:num]
		# Select files (paths/indices) for the batch
		Input_shuffle = []
		# Input2_shuffle = []

		out_shuffle = []

		# Read in each input, perform preprocessing and get labels
		for i in idx:
			img = cv2.resize(cv2.imread(files[i]),(256, 256))
			Input_shuffle += [(img+ (noise_factor[i] * np.random.normal(loc=0.0, scale=1.0, size=img.shape)))/255.]
			# Input2_shuffle += [cv2.resize(cv2.imread(files[i][1]),(64,64))/255.]
			out_shuffle += [img/255.]
		# Return a tuple of (input,output) to feed the network
		batch_x = np.array( Input_shuffle )
		# batch_x2 = np.array( Input2_shuffle )
		batch_y = np.array( out_shuffle )

		yield( batch_x, batch_y )


n = len(x_train)
n2  = len(x_test)

checkpoint = EpochCheckpoint('backups/', every=1,startAt=0)
model.fit_generator(image_generator(x_train,batch_size = batch_size),
					validation_data=image_generator(x_test,batch_size = batch_size)
					,validation_steps= n2 // batch_size,steps_per_epoch=n // batch_size,
					 epochs=10,callbacks=[checkpoint],verbose=1)



# #for visualization
# class Images(Callback):
#       def on_epoch_end(self, epoch, logs):
#             indices = np.random.randint(self.validation_data[0].shape[0], size=8)
#             test_data = self.validation_data[0][indices]
#             pred_data = self.model.predict(test_data)
#             run.history.row.update({
#                   "examples": [
#                         wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
#                         for i, data in enumerate(test_data)]
#             })

# # checkpoint = EpochCheckpoint('backups/', every=1,startAt=0)
# model.fit(x_train_noisy, x_train,
#                 epochs=config.epochs,
#                 validation_data=(x_test_noisy, x_test), callbacks=[Images()])


model.save("auto-denoise.h5")




