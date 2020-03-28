import cv2
import os

def getGabor(img, filters):
  res = cv2.filter2D(img, cv2.CV_8UC4, filters) # 2D滤波函数 kern为滤波模板
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


filters = cv2.getGaborKernel((11, 11), 11, 2.670353755551324, 11, 1.94, -1.5707963267948966)

# 读取文件
imgfiles = []
for root, dirs, files in os.walk("./img"):
  imgfiles = files

# 创建sift及flann初始化
# 创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create()
# 创建FLANN匹配对象
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# 遍历并缓存处理
cache = {}
for file in imgfiles:
  image = cv2.imread('./img/' + file, cv2.IMREAD_GRAYSCALE)
  cache[file] = {}
  cache[file]["kp"], cache[file]["des"] = sift.detectAndCompute(getGabor(image, filters), None)


# 遍历
for file1 in imgfiles:
  split_file_name1 = file1.split("_")
  img1kp = cache[file1]["kp"]
  img1des = cache[file1]["des"]

  for file2 in imgfiles:
    split_file_name2 = file2.split("_")
    kp2 = cache[file2]["kp"]
    des2 = cache[file2]["des"]
    matches = flann.knnMatch(img1des, des2, k=2)  # 匹配特征点，为了删除匹配点，指定k=2，对样本图每个特征点，返回两个匹配
    (matchNum, matchesMask) = getMatchNum(matches, 0.8)  # 通过比率条件，计算匹配度
    matchRatio = matchNum * 100 / len(matches)
    print(str(split_file_name1[0] == split_file_name2[0]) + "," + str(matchRatio) + "," + split_file_name1[0] + "," + split_file_name2[0])
