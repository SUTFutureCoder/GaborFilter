import numpy as np
import cv2

src_f = 0
kernel_size = 11
pos_sigma = 5
pos_lm = kernel_size - 2
pos_th = 0
pos_gam = 100
pos_psi = 90

def Process():
  sig = pos_sigma
  lm = pos_lm + 2
  th = pos_th * np.pi / 180.
  gm = pos_gam / 100.
  ps = (pos_psi - 180) * np.pi / 180
  print('kern_size=' + str(kernel_size) + ', sig=' + str(sig) + ', th=' + str(th) + ', lm=' + str(lm) + ', gm=' + str(gm) + ', ps=' + str(ps))
  kernel = cv2.getGaborKernel((kernel_size, kernel_size), sig, th, lm, gm, ps)
  kernelimg = kernel / 2. + 0.5
  global src_f
  dest = cv2.filter2D(image, cv2.CV_8UC4, kernel)
  cv2.imshow('Process window', dest)
  cv2.imshow('Kernel', cv2.resize(kernelimg, (kernel_size * 20, kernel_size * 20)))
  cv2.imshow('Mag', np.power(dest, 2))

def cb_sigma(pos):
  global pos_sigma
  if pos > 0:
    pos_sigma = pos
  else:
    pos_sigma = 1
  Process()

def cb_lm(pos):
  global pos_lm
  pos_lm = pos
  Process()

def cb_th(pos):
  global pos_th
  pos_th = pos
  Process()

def cb_gam(pos):
  global pos_gam
  pos_gam = pos
  Process()

def cb_psi(pos):
  global pos_psi
  pos_psi = pos
  Process()

if __name__ == "__main__":
  image = cv2.imread('BMP600/001_1.bmp', cv2.IMREAD_GRAYSCALE)
  cv2.imshow('Src', image)
  # global src_f
  src_f = np.array(image, dtype=np.float32)
  src_f /= 255.
  if not kernel_size % 2:
    kernel_size += 1

  cv2.namedWindow('Process window', 1)
  cv2.createTrackbar('Sigma', 'Process window', pos_sigma, kernel_size, cb_sigma)
  cv2.createTrackbar('Lambda', 'Process window', pos_lm, kernel_size - 2, cb_lm)
  cv2.createTrackbar('Theta', 'Process window', pos_th, 360, cb_th)
  cv2.createTrackbar('gamma', 'Process window', pos_psi, 300, cb_gam)
  cv2.createTrackbar('Psi', 'Process window', pos_psi, 360, cb_psi)
  Process()
  cv2.waitKey(0)
  cv2.destroyAllWindows()

