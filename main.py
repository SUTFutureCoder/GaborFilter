import cv2
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


def build_filters():
  filters = []
  ksize = [7, 9, 11, 13, 15, 17]
  lamda = np.pi / 2
  for theta in np.arange(0, np.pi, np.pi / 4):
    for K in range(6):
      kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.4045, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
      # kern = cv2.getGaborKernel((31, 31), 3.3, theta, 18.3, 4.5, 0.89, ktype=cv2.CV_32F)
      kern /= 1.5 * kern.sum()
      filters.append(kern)
  return filters


def process(img, filters):
  accum = np.zeros_like(img) # 初始化img一样大小的矩阵
  for kern in filters:
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kern) # 2D滤波函数 kern为滤波模板
    np.maximum(accum, fimg, accum) # 参数1与参数2逐位比较 取大者存入参数3 这里就是将文理特征显化更加明显
  return accum


def getGabor(img, filters):
  res = []
  for i in range(len(filters)):
    res1 = process(img, filters[i])
    res.append(np.asarray(res1))
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
image1 = cv2.imread('img/003_1.bmp', cv2.IMREAD_GRAYSCALE)
img1_gabor = getGabor(image1, filters)
image2 = cv2.imread('img/003_2.bmp', cv2.IMREAD_GRAYSCALE)
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


img1kp = [0] * 32
img1des = [0] * 32
for res1 in range(len(img1_gabor)):
  plt.figure('gabor', figsize=(8, 8))
  plt.subplot(121)
  plt.title('filt_imag')
  # pl.subplot(8, 6, res1 + 1)
  kp1, des1 = sift.detectAndCompute(img1_gabor[res1], None)  # 提取图片特征
  img1kp[res1] = kp1
  img1des[res1] = des1

  img1 = cv2.drawKeypoints(img1_gabor[res1], kp1, image1, color=(255, 0, 255))
  plt.imshow(img1)
  plt.show()
  # pl.imshow(img1, cmap='gray')

img2kp = [0] * 32
img2des = [0] * 32
comparisonImageList = []
for res2 in range(len(img2_gabor)):
  # pl.subplot(8, 6, len(img1_gabor) + res2 + 1)
  if img1des[res2] is 0 or img1_gabor[res2] is 0:
    continue
  kp2, des2 = sift.detectAndCompute(img2_gabor[res2], None)  # 提取对比图片的特征
  if des2 is None:
    continue
  try:
    matches = flann.knnMatch(img1des[res2], des2, k=2)  # 匹配特征点，为了删除匹配点，指定k=2，对样本图每个特征点，返回两个匹配
  except BaseException:
    continue
  (matchNum, matchesMask) = getMatchNum(matches, 0.9)  # 通过比率条件，计算匹配度
  matchRatio = matchNum * 100 / len(matches)
  print(str(res2)+" "+str(matchRatio) + "\n")
  drawParams = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matchesMask,
                    flags=0)
  comparisonImage = cv2.drawMatchesKnn(img1_gabor[res2], img1kp[res2], img2_gabor[res2], kp2, matches, None,
                                       **drawParams)
  comparisonImageList.append((comparisonImage, matchRatio))

  img2kp[res2] = kp2
  img2des[res2] = des2

  img2 = cv2.drawKeypoints(img2_gabor[res2], kp2, image2, color=(255, 0, 255))
  # pl.imshow(img2, cmap='gray')

# pl.show()
