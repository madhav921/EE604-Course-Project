import numpy as np
import cv2
import sys

def get_index(s):
    for i in range(len(s)):
        if s[i]>='0' and s[i]<='9':
            return int(s[i])

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    a=int((size-1)/2)
    for i in range (size):
      for j in range (size):
        kernel[i][j]=np.exp(-((i-a)**2+(j-a)**2)/(2*(sigma**2)))/(2*(np.pi)*(sigma**2))
    return kernel


def dilate(img, k):
    k=k//2
    temp = img
    y = img.shape[0]
    x = img.shape[1]
    for i in range(k,x-k):
        for j in range(k,y-k):
            val = 0
            for a in range(i-k,i+k+1):
                for b in range(j-k, j+k+1):
#             for a in range(max(0,i-k),min(x,i+k+1)):
#                 for b in range(max(0,j-k),min(y,j+k+1)):
                    val = max(val,img[b][a])
            temp[j][i] = val
    return temp


def median_blur(img, size):
    Hi, Wi = img.shape
    out = img
    Hk = size//2
    Wk = size//2
    for i in range(Wk,Wi-Wk):
        for j in range(Hk, Hi-Hk):
            wind = []
            for a in range(i-Wk,i+Wk+1):
                for b in range(j-Hk, j+Hk+1):
                    wind.append(img[b][a])
            wind.sort()
            out[j][i] = wind[(size**2)//2]
            wind.clear()
    out = out.astype(np.uint8)
    return out


def avg_conv(img, size):
    for p in range(3):
        image = img[:,:,p]
    #     kernel = gaussian_kernel(size, sigma)
        Hi, Wi = image.shape
        Hk = size
        Wk = size
        out = np.zeros((Hi, Wi))
        pad_width0 = Hk // 2
        pad_width1 = Wk // 2
        pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
        padded = np.pad(image, pad_width, mode='edge')
        for i in range(Hi):
            for j in range(Wi):
                multi = padded[i:i+Hk , j:j+Wk]
                k = (multi).sum()
                k /= Hk*Wk
                out[i][j] = int(k)
        out = out.astype(np.uint8)
        img[:,:,p] = out
    return img
    

def absolute_diff(img1,img2):
    out = img1
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            out[i][j] = (np.absolute(int(img1[i][j])-int(img2[i][j]))).clip(0,255)
    return out


def shadow_remove(img):
    for i in range(3):
        plane = img[:,:,i]
#         dilated_img = dilate(plane, 5)
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = median_blur(dilated_img, 3)
        diff_img = absolute_diff(plane, bg_img)
        diff_img = 255 - diff_img
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        img[:,:,i] = norm_img
    return img

def im1_additional(test1):
  img = test1
  for p in range(3):
      test1 = img[:,:,p]
      for i in range(test1.shape[0]):
          for j in range(test1.shape[1]):
              k = ((150)/(j+1))+30
              if test1[i][j]>100:
                  test1[i][j]-=int(k)
      img[:,:,p] = test1
  return test1

def overlap_imgs(img1,img2,alpha):
  for p in range(3):
      t1 = img1[:,:,p]
      t2 = img2
      for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            t1[i][j] = int(alpha*t1[i][j] + (1-alpha)*t2[i][j])
      img1[:,:,p] = t1
  return img1

def fourier_shadow(img):
  img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  Hi, Wi = img1.shape
  blur1 = cv2.blur(img1, [45, Hi])
  fourier = np.fft.fft2(blur1)
  fshift = np.fft.fftshift(fourier)
  mask = np.zeros(blur1.shape, np.uint8)
  # mask = disc(mask,mask.shape[1]//2, mask.shape[0]//2, 100 , 255)
  cv2.circle(mask, center = (mask.shape[1]//2, mask.shape[0]//2),radius =  100, color = 255, thickness = -1)
  mask = mask/255
  fshift = fshift*mask
  inv_shift = np.fft.ifftshift(fshift)
  img_back = np.real(np.fft.ifft2(inv_shift))
  # print(img_back.shape)
  # temp = [img_back, img_back, img_back]
  # sub_img = cv2.merge(temp)
  # fimg1 = im1 - sub_img + 240
  # fimg1 = 255 - absolute_diff(img1,img_back)+100
  fimg1 = img1 - img_back + 240
  fimg1 = np.real(fimg1).clip(0, 255)
  return fimg1


s = sys.argv[1]
img = cv2.imread(s)
k = get_index(s)

fimg = img


if k==1:
  f1 = shadow_remove(img)
  fimg = im1_additional(f1)
  # fimg = fourier_shadow(img)
else:
  fimg = fourier_shadow(img)
  

cv2.imwrite('cleared-gutter.jpg', fimg)

