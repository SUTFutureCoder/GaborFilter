
def getMatchNum(matches, ratio):
  matchesMask=[[0, 0] for i in range(len(matches))]
  matchNum = 0
  for i, (m, n) in enumerate(matches):
    if m.distance < ratio * n.distance: # 将距离小于ratio的匹配点筛选出来
      matchesMask[i] = [1, 0]
      matchNum += 1
  return matchNum

path=''
