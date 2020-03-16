import cv2

img1 = cv2.imread('imgs/WechatIMG100.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imgs/WechatIMG101.jpeg', cv2.IMREAD_GRAYSCALE)

# 获取特征提取器对象
orb = cv2.ORB_create()

# 检测关键点和特征描述
keypoint1, desc1 = orb.detectAndCompute(img1, None)
keypoint2, desc2 = orb.detectAndCompute(img2, None)
'''
keypoint 关键点列表
desc 检测到的特征的局部图列表
'''

# 获得knn检测器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(desc1, desc2, k=1)
'''
knn匹配可以返回k个最佳的匹配项
bf返回所有的匹配项
'''
# 画出匹配结果
img3 = cv2.drawMatchesKnn(img1, keypoint1, img2, keypoint2, matches, img2, flags=2)

# 缩放
scale_percent = 15
width = int(img3.shape[1] * scale_percent / 100)
height = int(img3.shape[0] * scale_percent / 100)
dim = (width, height)

img3 = cv2.resize(img3, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("matches", img3)
cv2.waitKey()
cv2.destroyAllWindows()

# 完全不准
