import numpy as np
import cv2
import sys
# from datetime import datetime

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


def joint_bilateral_upsample(ref_img, down_img, diameter, sigma_i, sigma_s):
    new_image = np.zeros((ref_img.shape[0],ref_img.shape[1]))
    kernel = gaussian_kernel(diameter,sigma_s)
    gauss_intensities = gauss_intensity(sigma_i)
    h1 = ref_img.shape[0]
    w1 = ref_img.shape[1]
    for row in range(h1):
        for col in range(w1):
            wp_total = 0
            conv_val = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x = (row - (diameter/2 - k))%h1
                    n_y = (col - (diameter/2 - l))%w1
                    gi = gauss_intensities[int(ref_img[int(n_x)][int(n_y)]) - int(ref_img[row][col])]
                    gi = gaussian(int(ref_img[int(n_x)][int(n_y)]) - int(ref_img[row][col]), sigma_i)
                    gs = kernel[k][l]
                    gs = gaussian(int(down_img[int(n_x)][int(n_y)]) - int(down_img[row][col]), sigma_s)
                    wp = gi * gs
                    conv_val += (down_img[int(n_x)][int(n_y)] * wp)
                    wp_total += wp
            conv_val = int(conv_val / wp_total)
            new_image[row][col] = int(np.round(conv_val))
    return new_image


def copy_upsample(img, size):
    up_img = np.zeros((int(size[0]), int(size[1])))
    fac = size[1]/img.shape[1]
    for i in range(size[0]):
        for j in range(size[1]):
            up_img[i][j] = img[int(i/fac)][int(j/fac)]
    return up_img


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

def swap02(img):
    red = img[:,:,2].copy()
    blue = img[:,:,0].copy()
    img[:,:,0] = red
    img[:,:,2] = blue
    return img

y = cv2.imread(sys.argv[1],0)
cb = cv2.imread(sys.argv[2],0)
cr = cv2.imread(sys.argv[3],0)

# start = datetime.now()

cb_ = copy_upsample(cb, y.shape)
cb_up = joint_bilateral_upsample(y, cb_, 3, 10, 10).clip(0,255)
cr_ = copy_upsample(cr, y.shape)
cr_up = joint_bilateral_upsample(y, cr_, 3, 10, 10).clip(0,255)

res = np.dstack((y,cb_up,cr_up)).astype('int64')

# end = datetime.now()
# td = (end - start).total_seconds()
# print(td//60, td%60)

res2 = ycbcr2rgb(res)
output_name = "flyingelephant.jpg"
temp2 = swap02(res2)
cv2.imwrite(output_name, temp2)

