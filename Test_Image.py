from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model('ship_detection_model.h5')

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

  def __init__(self):
    self.classes_names = {0: 'no-ship', 1: 'ship'}

  def draw_imgClass(self, img, classCode):
    print('Class : ', self.classes_names[classCode])
    plt.imshow(img, cmap='gray')
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
      NO_noShip += 1
    else:
      NO_ship += 1

def TestFunction(ImgPath):

    def formatNumber(num):
        if num % 1 == 0:
            return int(num)
        else:
            return num

    objDraw=cDraw()
    test_image = imread(ImgPath)
    test_d = test_image.copy()

    test_image = np.asarray(test_image, dtype=np.float32)
    test_image = test_image/255
    
    test_image = np.expand_dims(test_image, 0)
    result = model.predict(test_image)
    print(result[0])
    if result[0][0] < result[0][1]:
        class_path = "ship"
        print("Detected Result: Ship detected")
        colorRec= (0, 255, 0)
    elif result[0][0] == result[0][1]:
        class_path = "ship like"
        print("Detected Result: Ship like detected but Not Ship")
        colorRec = (0, 255, 0)
    else:
        class_path = "no ship"
        print("Detected Result: No Ship detected")
        colorRec= (255, 0, 0)
    imshow(objDraw.Draw_Rec(test_d, class_path, colorRec))

TestFunction(r'C:/Users/user/PycharmProjects/Ship Localization/Ships_Dataset/shipsnet/1__20160820_233143_0c53__-122.32816455984435_37.73917492083411.png')
TestFunction(r'C:/Users/user/PycharmProjects/Ship Localization/Ships_Dataset/shipsnet/0__20161116_180802_0e14__-122.36491377623285_37.793700719839684.png')
TestFunction(r'C:/Users/user/PycharmProjects/Ship Localization/Ships_Dataset/shipsnet/0__20161102_180658_0e26__-122.32843644745485_37.73923054907409.png')
TestFunction(r'C:/Users/user/PycharmProjects/Ship Localization/Ships_Dataset/shipsnet/0__20170901_181521_0e14__-122.56328572845645_37.73114111735287.png')
