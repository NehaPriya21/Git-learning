import numpy as np
import matplotlib.pyplot as plt
import cv


# Load image then grayscale
image = cv2.imread('open.jpg')

plt.figure(figsize=(10, 10))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(gray, 100, 200,apertureSize = 3)

plt.imshow(edge)
plt.show()


lines = cv2.HoughLinesP(edge,1,np.pi/180,240,minLineLength=1,maxLineGap=1)

for line in lines:
    x,y,w,h=line[0]
    cv2.line(image,(x,y),(w,h),(0,255,0),2)
    
plt.subplot(1, 1, 1)
plt.title(" Corners")
plt.imshow(image)
plt.show()
    


# In[103]:


len(lines)  


# In[104]:


lines


# In[105]:


lines[1:2]


# In[106]:


# Load image then grayscale
image1 = cv2.imread('op.jpg')

plt.figure(figsize=(10, 10))

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

edge1 = cv2.Canny(gray1, 100, 200,apertureSize = 3)

plt.imshow(edge1)
plt.show()


# In[107]:


lines1 = cv2.HoughLinesP(edge1,1,np.pi/180,240,minLineLength=150,maxLineGap=10)
for line in lines1:
    x1,y1,x2,y2 = line[0]
    cv2.line(image1,(x1,y1),(x2,y2),(0,255,0),2)
    
plt.subplot(1, 1, 1)
plt.title(" Corners")
plt.imshow(image1)
plt.show()


# In[108]:


len(lines1)  


# In[109]:


lines1


# In[114]:


# Load image then grayscale
image2 = cv2.imread('op2.jpg')

plt.figure(figsize=(10, 10))

gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

edge2 = cv2.Canny(gray2, 100, 200,apertureSize = 3)

plt.imshow(edge2)
plt.show()


# In[120]:


lines2 = cv2.HoughLinesP(edge2,1,np.pi/180,100,minLineLength=150,maxLineGap=10)

for line in lines2:
    
    x1,y1,x2,y2 = line[0]
    cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),2)
    
plt.subplot(1, 1, 1)
plt.title(" Corners")
plt.imshow(image2)
plt.show()


# In[121]:


len(lines2)


# In[122]:


lines2


# In[124]:


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

