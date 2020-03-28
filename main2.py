import cv2
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


def build_filters():
  # kern = cv2.getGaborKernel((21, 21), 5, 2.949606435870417, 13, 3.0, -1.5707963267948966)
  # kern = cv2.getGaborKernel((5, 5), 5, 2.6179938779914944, 5, 3.0, -0.10471975511965977)
  kern = cv2.getGaborKernel((11, 11), 11, 2.670353755551324, 11, 1.94, -1.5707963267948966)
  return kern


def process(img, filters):
  # src_f = np.array(img, dtype=np.float32)
  # src_f /= 255.
  fimg = cv2.filter2D(img, cv2.CV_8UC4, filters) # 2D滤波函数 kern为滤波模板
  return fimg


def getGabor(img, filters):
  res = process(img, filters)
  return res


def getMatchNum(matches, ratio):
  '''返回特征点匹配数量和匹配掩码'''
  matchesMask = [[0, 0] for i in range(len(matches))]
  matchNum = 0
  for i, (m, n) in enumerate(matches):
    if m.distance < ratio * n.distance:  # 将距离比率小于ratio的匹配点筛选出来
      matchesMask[i] = [1, 0]
      matchNum += 1
  return (matchNum, matchesMask)


filters = build_filters()
image1 = cv2.imread('img/001_3.bmp', cv2.IMREAD_GRAYSCALE)
img1_gabor = getGabor(image1, filters)
image2 = cv2.imread('img/001_4.bmp', cv2.IMREAD_GRAYSCALE)
img2_gabor = getGabor(image2, filters)

# 创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create()
# 创建FLANN匹配对象
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# pl.figure(2)
# pl.figure(8)


plt.figure('gabor', figsize=(8, 8))
plt.subplot(121)
plt.title('filt_imag')
pl.subplot(2, 1, 1)
kp1, des1 = sift.detectAndCompute(img1_gabor, None)  # 提取图片特征
img1kp = kp1
img1des = des1

img1 = cv2.drawKeypoints(img1_gabor, kp1, image1, color=(255, 0, 255))
# plt.imshow(img1)
# plt.show()
pl.imshow(img1, cmap='gray')

comparisonImageList = []
pl.subplot(2, 1, 2)
# pl.subplot(8, 6, len(img1_gabor) + res2 + 1)
kp2, des2 = sift.detectAndCompute(img2_gabor, None)  # 提取对比图片的特征
matches = flann.knnMatch(img1des, des2, k=2)  # 匹配特征点，为了删除匹配点，指定k=2，对样本图每个特征点，返回两个匹配
(matchNum, matchesMask) = getMatchNum(matches, 0.8)  # 通过比率条件，计算匹配度
matchRatio = matchNum * 100 / len(matches)
print(" "+str(matchRatio) + "\n")
drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=0)
comparisonImage = cv2.drawMatchesKnn(img1_gabor, img1kp, img2_gabor, kp2, matches, None,
                                     **drawParams)
# comparisonImageList.append((comparisonImage, matchRatio))

img2kp = kp2
img2des = des2

img2 = cv2.drawKeypoints(img2_gabor, kp2, image2, color=(255, 0, 255))
pl.imshow(img2, cmap='gray')

pl.show()
