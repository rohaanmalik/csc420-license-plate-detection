import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


img1 = cv.imread('cornerpads.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('sifttest5.jpg',cv.IMREAD_GRAYSCALE) # trainImage
img2 = cv.resize(img2, (300, 100))
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m, n in matches:
    if m.distance < 0.9*n.distance:
        good.append([m])

corners = []
for items in good:
    (x2, y2) = kp2[items[0].trainIdx].pt
    corners.append([x2, y2])

kmeans = KMeans(n_clusters=4, random_state=0).fit(np.array(corners))
print(kmeans.cluster_centers_)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()
