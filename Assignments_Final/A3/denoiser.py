import numpy as np
import cv2
from skimage.color import rgb2lab, lab2rgb
# from datetime import datetime
import sys

def gaussian(x,sigma):
    return (np.exp(-(x**2)/(2*(sigma**2)))/(2*np.pi*(sigma**2)))

def getDist(i,j, centre):
    iDist = centre[0]-i
    jDist = centre[1]-j
    dist = np.sqrt(jDist**2 + iDist**2)
    return dist

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    a=int((size-1)/2)
    for i in range (size):
        for j in range (size):
            kernel[i][j]=np.exp(-((i-a)**2+(j-a)**2)/(2*(sigma**2)))/(2*(np.pi)*(sigma**2))
    return kernel

def gauss_intensity(sigma):
    arr = np.zeros(256)
    for i in range(256):
        arr[i] = np.exp(-(i**2)/(2*(sigma**2)))/(2*np.pi*(sigma**2))
    return arr


def bilateral_filter(img, diameter, sigma_i, sigma_s):
    new_image = img
    h = img.shape[0]
    w = img.shape[1]
    dist_kernel = gaussian_kernel(diameter,sigma_s)
    gauss_intensities = gauss_intensity(sigma_i)
    for i in range(1):
        image = img[:,:,i]
        for row in range(h):
            for col in range(w):
                # if not mask[row][col]:
                #     continue
                if row<155 and col<100:
                    continue
                if row>275 and row<323 and col<440:
                    continue
                if row>250 and row<275 and col>295 and col<440:
                    continue
                if row>230 and row<250 and col>330 and col<380:
                    continue
                if row>292 and row<345 and col>500 and col<527:
                    continue
                if row>249 and row<275 and col>190 and col<225:
                    continue
                if row>252 and row<275 and col<55:
                    continue
                if row>365 and row<395 and col>150 and col<300:
                    continue
                if row>350 and row<395 and ((col>150 and col<190) or (col>215 and col<245) or (col>275 and col<300)):
                    continue
                if row>325 and row<410 and col>380 and col<430:
                    continue
                if row>425:
                    continue
                if row>350 and col>480:
                    continue
                wp_total = 0
                conv_val = 0
#                 multi = padded[row-diameter/2:row+diameter/2 , col-diameter/2:col+diameter/2]
                for k in range(diameter):
                  for l in range(diameter):
                      n_x = (row - (diameter/2 - k))%h
                      n_y = (col - (diameter/2 - l))%w
                      gi = gaussian(int(image[int(n_x)][int(n_y)]) - int(image[row][col]), sigma_i)
#                       gi = gauss_intensities[int(image[int(n_x)][int(n_y)]) - int(image[row][col])]
                      gs = dist_kernel[k][l]
                      wp = gi * gs
                      conv_val += (image[int(n_x)][int(n_y)] * wp)
                      wp_total += wp
                conv_val = conv_val // wp_total
                new_image[row][col][i] = int(np.round(conv_val))
    return new_image


def bilateral_filter1(img, diameter, sigma_i, sigma_s):
    new_image = img
    h = img.shape[0]
    w = img.shape[1]
    dist_kernel = gaussian_kernel(diameter,sigma_s)
    gauss_intensities = gauss_intensity(sigma_i)
    for i in range(1):
        image = img[:,:,i]
        for row in range(h):
            for col in range(w):
                # if not mask[row][col]:
                #     continue
                if row<155 and col<100 and not(row>47 and row<60 and col<20) and not(row>105 and row<125 and col>50 and col<65):
                    continue
                if row>275 and col<385:
                    continue
                if row>250 and row<275 and col>295 and col<440:
                    continue
                if row>230 and row<250 and col>330 and col<380:
                    continue
                if row>310 and col>440:
                    continue
                if row>292 and row<345 and col>500 and col<527:
                    continue
                if row>249 and row<275 and col>190 and col<225:
                    continue
                if row>252 and row<275 and col<55:
                    continue
                if row>365 and row<395 and col>150 and col<300:
                    continue
                if row>350 and row<395 and ((col>150 and col<190) or (col>215 and col<245) or (col>275 and col<300)):
                    continue
                if row>325 and row<410 and col>380 and col<430:
                    continue
                if row>425:
                    continue
                if row>350 and col>480:
                    continue
                wp_total = 0
                conv_val = 0
#                 multi = padded[row-diameter/2:row+diameter/2 , col-diameter/2:col+diameter/2]
                for k in range(diameter):
                  for l in range(diameter):
                      n_x = (row - (diameter/2 - k))%h
                      n_y = (col - (diameter/2 - l))%w
                      gi = gaussian(int(image[int(n_x)][int(n_y)]) - int(image[row][col]), sigma_i)
#                       gi = gauss_intensities[int(image[int(n_x)][int(n_y)]) - int(image[row][col])]
                      gs = dist_kernel[k][l]
                      wp = gi * gs
                      conv_val += (int(image[int(n_x)][int(n_y)]) * wp)
                      wp_total += wp
                conv_val = conv_val // wp_total
                new_image[row][col][i] = int(np.round(conv_val))
    return new_image

def swap02(img):
    ch1 = img[:,:,0].copy()
    ch2 = img[:,:,2].copy()
    img[:,:,2] = ch1
    img[:,:,0] = ch2
    return img


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def bilateral_filter2(img, diameter, sigma_i, sigma_s):
    new_image = img
    h = img.shape[0]
    w = img.shape[1]
    dist_kernel = gaussian_kernel(diameter,sigma_s)
    gauss_intensities = gauss_intensity(sigma_i)
    m = 750/(420-200)
    for i in range(1):
        image = img[:,:,i]
        for row in range(h):
            if row>420:
                break
            for col in range(w):
                wp_total = 0
                conv_val = 0
#                 multi = padded[row-diameter/2:row+diameter/2 , col-diameter/2:col+diameter/2]
                for k in range(diameter):
                  for l in range(diameter):
                      n_x = (row - (diameter/2 - k))%h
                      n_y = (col - (diameter/2 - l))%w
                      gi = gaussian(int(image[int(n_x)][int(n_y)]) - int(image[row][col]), sigma_i)
#                       gi = gauss_intensities[int(image[int(n_x)][int(n_y)]) - int(image[row][col])]
                      gs = dist_kernel[k][l]
                      wp = gi * gs
                      conv_val += (image[int(n_x)][int(n_y)] * wp)
                      wp_total += wp
                conv_val = conv_val // wp_total
                new_image[row][col][i] = int(np.round(conv_val))
    return new_image

def get_mask(img,low,high):
    mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0]>low and img[i][j][0]<high:
                mask[i][j] = 1
    return mask

def get_index(s):
    for i in range(len(s)):
        if s[i]>='0' and s[i]<='9':
            return int(s[i])

s = sys.argv[1]
img = cv2.imread(s)
k = get_index(s)

# start = datetime.now()

if k==1:
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    mask = get_mask(hsv,98,103)
    img_own_lab1 = rgb2lab(img1)
    image_own2 = bilateral_filter(img_own_lab1, 7, 13, 2)
    image_own2 = bilateral_filter1(image_own2, 7, 4, 1)
    img1_out = lab2rgb(image_own2)

    img1_out *= int(255/img1_out.max())
    out = img1_out.astype('int64')

else:
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_own_lab2 = rgb2lab(img2)
    image_own = bilateral_filter2(img_own_lab2, 7, 8, 2)
    img2_out = lab2rgb(image_own)

    img2_out *= int(255/img2_out.max())
    out = img2_out.astype('int64')

output_name = "denoised.jpg"
temp = swap02(out)
cv2.imwrite(output_name, temp)

# end = datetime.now()
# td = (end - start).total_seconds()
# print(td//60, td%60)