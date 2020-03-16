import cv2
from matplotlib import pyplot as plt


def get_match_num(matches, ratio):
  """返回特征点匹配数量和匹配掩码"""
  matches_mask = [[0, 0] for i in range(len(matches))]
  match_num = 0
  for i, (m, n) in enumerate(matches):
    if m.distance < ratio * n.distance:
      matches_mask[i] = [1, 0]
      match_num += 1
  return match_num, matches_mask


img1 = cv2.imread('imgs/WechatIMG100.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imgs/WechatIMG101.jpeg', cv2.IMREAD_GRAYSCALE)

# 创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create()
# 创建FLANN匹配对象
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matches = flann.knnMatch(des1, des2, k=2) # 匹配特征点，为了筛选匹配点，指定k为2，为样本图的每个特征点，返回两个匹配
match_num, matches_mask = get_match_num(matches, 0.7) # 通过比率条件，计算出匹配程度
match_ratio = match_num * 100 / len(matches)

draw_params = dict(
  matchColor=(0, 255, 0),
  singlePointColor=(255, 0, 0),
  matchesMask=matches_mask,
  flags=0
)
comparison_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

print(match_ratio)

scale_percent = 15
width = int(comparison_image.shape[1] * scale_percent / 100)
height = int(comparison_image.shape[0] * scale_percent / 100)
dim = (width, height)
img3 = cv2.resize(comparison_image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('matches', img3)
cv2.waitKey()
cv2.destroyAllWindows()



