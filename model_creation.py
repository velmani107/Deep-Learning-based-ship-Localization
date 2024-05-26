import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import os
import random
from tqdm import tqdm

import imgaug.augmenters as iaa

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
# from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

class cCreate_DataSets:

    def __init__(self, DataPath):
        self.DataPath = DataPath
        self.categories = {'no-ship': 0,  'ship': 1}
        # Augmentation for balancing class 1
        self.seq_class1_rotate = iaa.Sequential([ iaa.Affine(rotate=(-20, 20))])  # rotate
        self.seq_class1_HS = iaa.Sequential([iaa.AddToHueAndSaturation(value=(-20, 20))])  # Change hue and saturation
        self.seq_class1_Scale = iaa.Sequential([iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})])  # Scale in x and y direction

    def augment_image(self, image, class_num):
        if class_num == 1:
            temp_ro=image.copy()
            temp_HS=image.copy()
            temp_sc=image.copy()

            temp_ro= self.seq_class1_rotate.augment_image(temp_ro)
            temp_HS= self.seq_class1_HS.augment_image(temp_HS)
            temp_sc= self.seq_class1_Scale.augment_image(temp_sc)

            return temp_ro,temp_HS,temp_sc
        else:
            return image

    def Create_Raw_Image(self, arrayDataSet):
        for class_path, encode in self.categories.items():
            CAT_PATH = os.path.join(self.DataPath, class_path)
            for imgs in tqdm(os.listdir(CAT_PATH)):
                try:
                    img = cv2.imread(os.path.join(CAT_PATH, imgs))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if encode == 1 :
                      img_ro,img_hs,img_sc = self.augment_image(img, encode)

                      img_ro=img_ro.astype('float32')/255.0  # Normalization
                      img_hs=img_hs.astype('float32')/255.0  # Normalization
                      img_sc=img_sc.astype('float32')/255.0  # Normalization

                      arrayDataSet.append([img_ro, encode])
                      arrayDataSet.append([img_hs, encode])
                      arrayDataSet.append([img_sc, encode])

                    img = img.astype('float32')/255.0  # Normalization
                    arrayDataSet.append([img, encode])
                except Exception as e:
                    print(e)

        return arrayDataSet

    def Create_GaussianProc_Image(self, arrayDataSet):
        for class_path, encode in self.categories.items():
            CAT_PATH = os.path.join(self.DataPath, class_path)
            for imgs in tqdm(os.listdir(CAT_PATH)):
                try:
                    img = cv2.imread(os.path.join(CAT_PATH, imgs))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    blurred = cv2.GaussianBlur(img, (3, 3), 0)  # Smoothing Edges
                    if encode == 1:
                      img_ro, img_hs, img_sc = self.augment_image(blurred, encode)

                      img_ro=img_ro.astype('float32')/255.0  # Normalization
                      img_hs=img_hs.astype('float32')/255.0  # Normalization
                      img_sc=img_sc.astype('float32')/255.0  # Normalization

                      arrayDataSet.append([img_ro, encode])
                      arrayDataSet.append([img_hs, encode])
                      arrayDataSet.append([img_sc, encode])

                    blurred = blurred.astype('float32')/255.0  # Normalization
                    arrayDataSet.append([blurred, encode])
                except Exception as e:
                    print(e)
        return arrayDataSet

def imread(path):
    imbgr = cv2.imread(path)
    return cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)  # retern img rgb


def imshow(img, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.show()


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb2bin(img):
    t, imbin = cv2.threshold(rgb2gray(img), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return imbin

class cDraw:

  def __init__(self ):
    self.classes_names = {0: 'no-ship', 1: 'ship'}

  def draw_imgClass(self, img, classCode):
    print('Class : ', self.classes_names[classCode])
    plt.imshow(img, cmap='gray')
    plt.show()

  def draw_history(self, history):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].plot(history.history['accuracy'], label='Train Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')

    ax[0].legend(loc='upper right')
    ax[1].legend(loc='lower right')
    plt.show()

  def Draw_Rec(self, img, class_path, colorRec):
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Smoothing Edges
    components = cv2.connectedComponentsWithStats(
      rgb2bin(img), connectivity=4)
    (nLabels, labels, stats, centroids) = components
    centroids = centroids.astype(int)
    img_d = img.copy()

    for i in range(0, 1):
        x, y, w, h, area = stats[i]
        cv2.rectangle(img_d, (x, y), (x+w-1, y+h-1), colorRec, 1)
        cv2.putText(img_d, class_path, (x+1, y+h-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colorRec, 1)
    return img_d

def NO_classes(df):
  NO_ship=0
  NO_noShip=0
  for d, i in df:
    if i == 0:
      NO_noShip +=1
    else:
      NO_ship +=1

  print(f' Total Images = {len(df) }')
  print(f' Number of Images class No ship = {NO_noShip}')
  print(f' Number of Images class ship = {NO_ship}')

def TestFunction(ImgPath):
    objDraw=cDraw()
    test_image = imread(ImgPath)
    test_d =test_image.copy()

    test_image=np.asarray(test_image, dtype=np.float32)
    test_image=test_image/255

    test_image= np.expand_dims(test_image, 0)
    result=model.predict(test_image)

    if result[0][0] < result[0][1]:
        class_path = "ship"
        colorRec= (0, 255, 0)
    else:
        class_path = "no ship"
        colorRec= (255, 0, 0)

    imshow(objDraw.Draw_Rec(test_d, class_path, colorRec))

DATA_SET_PATH = r'E:/Arunai_CSE_Projects/Ship_Localization_with_Source_and_Destination/Dataset'

# create objects
objCreateData = cCreate_DataSets(DATA_SET_PATH)
objDraw = cDraw()

print(objCreateData.categories)

dataSet=[]
dataSet=objCreateData.Create_Raw_Image(dataSet)

print(NO_classes(dataSet))

clsCdoe = dataSet[500][1]
imgRaw = dataSet[500][0]

objDraw.draw_imgClass(imgRaw, clsCdoe)

# original img
clsCdoe = dataSet[3051][1]
imgRaw = dataSet[3051][0]
objDraw.draw_imgClass(imgRaw, clsCdoe)

# scaled img
clsCdoe = dataSet[3050][1]
imgRaw = dataSet[3050][0]
objDraw.draw_imgClass(imgRaw, clsCdoe)

# hue and saturation img
clsCdoe = dataSet[3049][1]
imgRaw = dataSet[3049][0]
objDraw.draw_imgClass(imgRaw, clsCdoe)

# rotate img
clsCdoe = dataSet[3048][1]
imgRaw = dataSet[3048][0]

objDraw.draw_imgClass(imgRaw, clsCdoe)

dataGauss=[]
dataGauss=objCreateData.Create_GaussianProc_Image(dataGauss)

print(NO_classes(dataGauss))

clsCdoe = dataGauss[500][1]
imgRaw = dataGauss[500][0]

objDraw.draw_imgClass(imgRaw, clsCdoe)

# original img
clsCdoe = dataGauss[3051][1]
imgRaw = dataGauss[3051][0]
objDraw.draw_imgClass(imgRaw, clsCdoe)

# scaled img
clsCdoe = dataGauss[3050][1]
imgRaw = dataGauss[3050][0]
objDraw.draw_imgClass(imgRaw, clsCdoe)

# hue and saturation img
clsCdoe = dataGauss[3049][1]
imgRaw = dataGauss[3049][0]
objDraw.draw_imgClass(imgRaw, clsCdoe)

random.shuffle(dataGauss)
X_Gauss=[]
Y_Gauss=[]
for feature, label in dataGauss:
    X_Gauss.append(feature)
    Y_Gauss.append(label)

xGauss_train, xGauss_test, yGauss_train, yGauss_test = train_test_split(X_Gauss, Y_Gauss, test_size=0.10, random_state=42)
xGauss_train, xGauss_valid, yGauss_train, yGauss_valid = train_test_split(xGauss_train, yGauss_train, test_size=0.15, random_state=42)

print('train : ', len(yGauss_train))
for i in range(2):
  print(i, " -- ", yGauss_train.count(i))

print('valid : ', len(yGauss_valid))
for i in range(2):
  print(i, " -- ", yGauss_valid.count(i))

print('test : ', len(yGauss_test))
for i in range(2):
  print(i, " -- ", yGauss_test.count(i))

xGauss_train = np.array(xGauss_train)
yGauss_train = np.array(yGauss_train)

xGauss_valid = np.array(xGauss_valid)
yGauss_valid = np.array(yGauss_valid)

xGauss_test = np.array(xGauss_test)
yGauss_test = np.array(yGauss_test)

print(xGauss_train[0].shape)

yGauss_train = to_categorical(yGauss_train)
yGauss_valid = to_categorical(yGauss_valid)

print(yGauss_train[1].shape)
print(yGauss_train[1])

model= keras.Sequential([
    Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(80, 80, 3)),
    BatchNormalization(),
    MaxPool2D(2, padding='same'),

    Conv2D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPool2D(2),
    Dropout(0.3),

    Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(2),
    Dropout(0.25),

    Conv2D(512, kernel_size=3, activation='relu', name='last_conv'),
    BatchNormalization(),
    MaxPool2D(2),
    Dropout(0.25),

    Flatten(),
    Dense(128),
    Dropout(0.2),
    Dense(64),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.summary()

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Recall()])

hist = model.fit(x=xGauss_train, y=yGauss_train, batch_size=64, epochs=15,
                 validation_data=(xGauss_valid, yGauss_valid),
                 validation_batch_size=64)

objDraw.draw_history(hist)

model.save('ship_detection_model.h5')

y_pred = model.predict(xGauss_test)

y_pred_classes = np.argmax(y_pred, axis=1)
confusion_Matrix = confusion_matrix(yGauss_test, y_pred_classes)
sns.heatmap(confusion_Matrix, annot=True, cmap='Blues', fmt='.3g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print(classification_report(yGauss_test, y_pred_classes))

TestFunction(r'E:/Arunai_CSE_Projects/Ship_Localization_with_Source_and_Destination/Dataset/ship/ship_000090.png')

TestFunction(r'E:/Arunai_CSE_Projects/Ship_Localization_with_Source_and_Destination/Dataset/no-ship/no-ship_002000.png')
