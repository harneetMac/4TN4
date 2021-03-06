# -*- coding: utf-8 -*-
"""ANN vs CNN project2-4TN4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1igpayj9F9N1kFkSBGNhTtrihuNyyPgXL
"""

from google.colab import drive
drive.mount('/content/drive/')

!pip install -q keras
!pip install tensorflow
!pip install tflearn
!pip install tqdm

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

LR = 1e-3

MODEL_NAME = f'book-recognition-{LR}-2conv-basic.model'

IMG_SIZE = 50
DIR = '/content/drive/MyDrive/4TN4-Project2-Dataset'
nRowsRead = None

def create_train_data():
  training_data = []
  # training_data = np.empty((IMG_SIZE, IMG_SIZE))
  label_data = []

  for dirname, _, filenames in os.walk(DIR):
    for filename in filenames:
      path = os.path.join(dirname, filename)
      # print(path)

      if path.split(".")[-1] == "jpg":
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        # print(img.shape)
        # training_data.append([np.array(img)])
        training_data.append([img])
        # training_data = np.append(training_data, np.array(img), axis=0)

      elif path.split(".")[-1] == "csv":
        # print(path)
        df = pd.read_csv(path, delimiter=',', nrows = nRowsRead)
        # df.dataframeName = path.split(".")[-2].split("/")[-1]
        # print(df.dataframeName)
        nRow, nCol = df.shape
        # print(f'There are {nRow} rows and {nCol} columns')

        for i in range(nRow):
          # print(df.iloc[i, 1], df.iloc[i, 2])
          # label_data.append([np.array([np.array(df.iloc[i, 1])], [np.array(df.iloc[i, 2])])])
          # label_data.append([np.array(df.iloc[i, 1])])
          l = [df.iloc[i, 1]]
          # print(l)
          label_data.append(l)
  
  # print(training_data.shape)
  print(f'training data:', len(training_data))
  print(f'labels:', len(label_data))

  return training_data, label_data

train_data, label_data = create_train_data()

# save data into a file

np.save('train_data.npy', train_data)
np.save('label_data.npy', label_data)

# train_data = np.load("train_data.npy")
# label_data = np.load("label_data.npy")

book_names = label_data

label_data = np.array(label_data)
label_data = label_data[:- 32081]
print(label_data.shape)

train_data = np.array(train_data, dtype="float") / 255.0
train_data = train_data[:- 32081]
print(train_data.shape)

print(label_data)

print(train_data[0])

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
label_data = mlb.fit_transform(label_data)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
  print("{}. {}".format(i + 1, label))
  break

train = train_data
test = train_data[-20:]
print(train.shape)
test.shape

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# Y = label_data[:-100]
Y = np.arange(start=0,stop=train.shape[0],step=1)

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# test_y = label_data[-100:]
test_y = np.arange(start=train.shape[0]-test.shape[0],stop=train.shape[0],step=1)

a = np.arange(start=0,stop=400,step=1)
print(a)

print(X.shape)
print(Y.shape)
print(test_x.shape)
print(test_y.shape)

"""# Artificial Neural Network"""

#For Tensorboard
logdir = "/content/ann-logs/" 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

ann = models.Sequential([
        layers.Flatten(input_shape=(IMG_SIZE,IMG_SIZE,1)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(500, activation='softmax')    
    ])

ann.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.summary()

ann.fit(X, Y, epochs=20, callbacks=[tensorboard_callback])

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=ann-logs

def plot_sample(X, y, index, test=False):
  c= np.ravel(X[index])
  c = ((np.reshape(c, (50,50)))*255).astype('uint8')
  plt.imshow(c, cmap='gray')
  if not test:
    plt.xlabel(book_names[index])
  elif test:
    plt.xlabel(book_names[480+index])
  plt.show()

plot_sample(train_data, Y, 4)

ann.evaluate(test_x, test_y)

from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(test_x)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(test_y, y_pred_classes))

# Evaluating confusion matrix
res = tf.math.confusion_matrix(y_classes,test_y)
 
# Printing the result
for i in range(21):
  print('Confusion_matrix: ',np.argmax(res[-i]))

"""# CNN"""

#For Tensorboard
cnn_logdir = "/content/cnn-logs/" 
tensorboard_cnn_callback = tf.keras.callbacks.TensorBoard(log_dir=cnn_logdir)

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(500, activation='softmax')
])
cnn.summary()

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X, Y, epochs=20, callbacks=[tensorboard_cnn_callback])

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=cnn-logs

cnn.evaluate(test_x, test_y)

plot_sample(test_x, test_y, 4, test=True)

y_pred = cnn.predict(test_x)
y_pred[:5]

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]

test_y[:5]

plot_sample(test_x, test_y, 1, test=True)

book_names[y_classes[1]]

print("Classification Report: \n", classification_report(test_y, y_classes))

# Evaluating confusion matrix
res = tf.math.confusion_matrix(y_classes,test_y)
 
# Printing the result
for i in range(21):
  print('Confusion_matrix: ',np.argmax(res[-i]))