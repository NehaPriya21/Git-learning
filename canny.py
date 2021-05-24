import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("opencv.JPG", cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
edges= cv2.Canny(img,100,200)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny']
images = [img, lap, sobelX, sobelY, sobelCombined, edges]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

img = cv2.imread("opencv.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edge = cv2.Canny(img, 100, 200,apertureSize = 3)
lines = cv2.HoughLinesP(edge,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
  x1,y1,x2,y2 = line[0]
  cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2,)

cv2.imshow('img',img)
k = cv2.waitKey(0)
c=cv2.destroyAllWindows()

titles = ['image', 'edge']
images = [img, edge]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()




