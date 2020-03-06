from keras.models import Model
from keras.models import load_model

from keras.datasets import mnist
import numpy as np
import cv2
from data_load import data_load
import cv2


fname = 'val2017'
x_train,x_test = data_load(fname)

model = load_model('backups/epoch_10.h5')
print(model.summary())
def add_noise(x_train):
    noise_factor = 1
    # img = x_train.astype(np.float32)
    # row, col, _ = x_train.shape
    # gaussian = np.random.random((row, col, 1)).astype(np.float32)
    # gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    # gaussian_img = cv2.addWeighted(img, 0.75, 0.25 * gaussian, 0.25, 0)
    # gaussian_noise_imgs.append(gaussian_img)
    # gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    x_train_noisy = (x_train + (noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)))
    # print(x_train_noisy)
    # print(x_train)
    # cv2.imshow('noise', x_train_noisy.astype(np.uint8))
    # cv2.imshow('image', img.astype(np.uint8))
    # cv2.waitKey(0)

    # x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    # print(x_train_noisy)
    return x_train_noisy

i = 0
while(True):

  k = cv2.waitKey(0)
  if k == 27:         # wait for ESC key to exit
      cv2.destroyAllWindows()
      break

  test_img = cv2.resize(cv2.imread(x_test[i]),(256, 256))

  if k == 32:   # space bar
      pass
  test_img = add_noise(test_img)
  input_img = np.expand_dims(test_img/255, axis=0)

  output_img = model.predict([input_img])[0]*255.
  print(output_img.shape)
  cv2.imshow('input', test_img.astype(np.uint8))
  cv2.imshow('output', output_img.astype(np.uint8))
  i+=1




