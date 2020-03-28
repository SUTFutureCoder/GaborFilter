from skimage import io, filters, data
import cv2
import matplotlib.pyplot as plt

'''
https://www.cnblogs.com/denny402/p/5125253.html
'''

img = io.imread('img/002_4.bmp')



filt_real, filt_imag = filters.gabor(img, frequency=0.75)

# filt_real = filters.gaussian(filt_imag, sigma=0.1)

plt.figure('gabor', figsize=(8, 8))
plt.subplot(121)
plt.title('filt_imag')
plt.imshow(filt_imag, plt.gray())

plt.show()