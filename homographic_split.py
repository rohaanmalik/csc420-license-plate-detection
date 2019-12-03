import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def getHomographicMatrix(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x1, y1 = p1[i][0], p1[i][1]
        x2, y2 = p2[i][0], p2[i][1]
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)
    u,s,v = np.linalg.svd(A)
    L = v[-1, :] / v[-1, -1]
    h = L.reshape(3, 3)
    return h

def getPairPoint(M, p1):
    src = np.array([p1[0], p1[1], 1])
    dst = M @ src
    dstInd = np.array([dst[0]//dst[2], dst[1]//dst[2]])
    return dstInd.astype(int)


def mapToTemplate(x,y,M,src):
    pad = np.zeros((y,x)).astype('uint8')
    # srcy = src.shape[0]
    # srcx = src.shape[1]
    for i in range(x):
        for j in range(y):
            srcPoint = getPairPoint(M, np.array([i,j]))
            pad[j][i] = src[srcPoint[1]][srcPoint[0]]
    return pad

def projection(y,x,src, p2):
    p1 = np.array([[0, 0], [0, y], [x, 0], [x, y]])
    H = getHomographicMatrix(p1,p2)
    dst = mapToTemplate(x,y,H,src)
    return dst

def maxCont(array, n=None):
    curStart = None
    curPosition = 0
    isCont = False
    ret = []
    for i in range(len(array)):
        if curStart is None:
            if (array[i]) > 0:
                curStart = i
                curPosition = i
                isCont = True
        else:
            if(array[i]) > 0:
                curPosition = i
                if not isCont:
                    curStart = i
                    curPosition = i
                    isCont = True
            else:
                if isCont:
                    isCont = False
                    ret.append((curStart, curPosition, curPosition-curStart))
    if isCont:
        ret.append((curStart, curPosition, curPosition-curStart))

    if n is not None:
        sort_width = sorted(ret, key=lambda x: x[2], reverse=True)
        if len(sort_width) >= n:
            ret = sort_width[:n]
            ret = sorted(ret, key=lambda x: x[0])
    return ret

def imgThreshold(img, threshAlpha):
    thresh, ret = cv.threshold(img, np.mean(img) * threshAlpha, 1, cv.THRESH_BINARY)
    ret = ret.astype('float')
    ret = 1 - ret
    return ret

def imgVfilter(img, filterAlpha = 10):
    vfilter = np.sum(img, axis=1)
    vfilter = np.where(vfilter > filterAlpha, 1, 0)
    return vfilter


def imgHfilter(img, filterAlpha = 2):
    hfilter = np.sum(img, axis=0)
    hfilter = np.where(hfilter > filterAlpha, 1, 0)
    return hfilter

def numberVSplit(img, vfilter):
    vlimit = maxCont(vfilter, 1)[0]
    pad = np.zeros((vlimit[1] - vlimit[0], img.shape[1]))
    pad[:, :] = img[vlimit[0]:vlimit[1], :]
    return pad


def numberHSplit(img, hfilter, n = 10):
    hlimits = maxCont(hfilter, n)
    for i in range(len(hlimits)):
        hlimit = hlimits[i]
        number = np.zeros((img.shape[0], hlimit[1] - hlimit[0]))
        number[:, :] = img[:, hlimit[0]:hlimit[1]]
        number = (number * 255).astype('uint8')
        ratio = int(np.max(number.shape) // 20)
        number = cv.resize(number, (number.shape[1] // ratio, number.shape[0] // ratio))
        y = number.shape[0]
        x = number.shape[1]
        ystart = (28 - y) // 2
        xstart = (28 - x) // 2
        pad = np.zeros((28,28)).astype('uint8')
        pad[ystart:ystart+y,xstart:xstart+x] = number
        pad = 255 - pad
        cv.imwrite("number" + str(i) + ".png", pad)


# img = cv.imread("test.jpg", cv.IMREAD_GRAYSCALE)
# p2 = np.array([[150,180], [150,255], [470,205], [470,295]])

# img = cv.imread("test4.jpg", cv.IMREAD_GRAYSCALE)
# p2 = np.array([[230,110], [220,150], [350,125], [340,175]])

img = cv.imread("test5.jpg", cv.IMREAD_GRAYSCALE)
p2 = np.array([[530,432], [520,496], [645,410], [630,468]])

# img = cv.imread("test6.png", cv.IMREAD_GRAYSCALE)
# p2 = np.array([[37,24], [32,125], [355,22], [345,120]])

# img = cv.imread("100.jpg", cv.IMREAD_GRAYSCALE)
# p2 = np.array([[241,190], [241,232], [377,190], [377,232]])

# img = cv.imread("5.jpg", cv.IMREAD_GRAYSCALE)
# p2 = np.array([[344,371], [344,438], [493,371], [493,438]])

ret = projection(100, 300, img, p2)
# cv.imshow("ret", ret)
# cv.waitKey(0)
# cv.destroyAllWindows()

cv.imwrite("projected.jpg", ret)
thresimg = imgThreshold(ret, 0.75)
vfilter = imgVfilter(thresimg)

vcut = numberVSplit(thresimg, vfilter)
hfilter = imgHfilter(vcut)

cv.imwrite("threshold.jpg", thresimg*255)
numberHSplit(vcut, hfilter, 10)


# hfilter = np.sum(vcut, axis=0)
# plt.plot(hfilter)
# plt.title('horizontal intensity')
# plt.ylabel('intensity')
# plt.xlabel('column')
# plt.show()