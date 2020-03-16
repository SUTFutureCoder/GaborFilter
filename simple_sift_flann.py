import cv2

'''SIT+FLANN'''
# 按照灰度图片读入
img1 = cv2.imread('imgs/WechatIMG100.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imgs/WechatIMG101.jpeg', cv2.IMREAD_GRAYSCALE)

# 创建sift检测器
sift = cv2.xfeatures2d.SIFT_create()
# 查找监测点和匹配符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
'''
keypoint是检测到的特征点的列表
descriptor是检测到特征的局部图像的列表
'''
# 获取flann匹配器
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
# 进行匹配
matches = flann.knnMatch(des1, des2, k=2)
# 准备空的掩膜 画好的匹配项
matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
  if m.distance < 0.7 * n.distance:
    matchesMask[i] = [1, 0]

drawPrams = dict(matchColor=(0, 255, 0),
                 singlePointColor=(255, 0, 0),
                 matchesMask=matchesMask,
                 flags=0)

# 匹配结果图片
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawPrams)

# 缩放
scale_percent = 15
width = int(img3.shape[1] * scale_percent / 100)
height = int(img3.shape[0] * scale_percent / 100)
dim = (width, height)
img3 = cv2.resize(img3, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('matches', img3)
cv2.waitKey()
cv2.destroyAllWindows()



