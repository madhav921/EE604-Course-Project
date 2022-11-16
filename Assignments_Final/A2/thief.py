import numpy as np
import cv2
import sys

def get_index(s):
    for i in range(len(s)):
        if s[i]>='0' and s[i]<='9':
            return int(s[i])


def log_transform(img):
    for layer in range(3):
        plane = img[:,:,layer]
        c = 255/(np.log(1 + 3*np.max(plane)))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k = plane[i][j]
                k = c * np.log(1 + plane[i][j])
                if(k>255):
                    k=255
                plane[i][j] = int(k)
        img[:,:,layer] = plane
    return img


def gamma_transform(img, gamma):
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    return gamma_corrected

def gray_patch(img, fac=2):
    for layer in range(3):
        plane = img[:,:,layer]
        mean = np.mean(plane)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k = plane[i][j]
                k/=fac*mean
                k*=255
                if(k>255):
                    k=255
                plane[i][j] = int(k)
        img[:,:,layer] = plane
    return img

def histogram_equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Wi, Hi = img.shape
    img_array = np.asarray(img)
    
    pdf = []
    cdf = []
    tr_map = []
    
    histogram_array = np.bincount(img_array.flatten(), minlength=256)
    
    num_pixels = 0
    for i in histogram_array:
        num_pixels += i
    
    for i in histogram_array:
        pdf.append(i/num_pixels)
    
    sum=0
    for i in range(256):
        sum += pdf[i]
        cdf.append(sum)
        tr_map.append(np.floor(255 * sum).astype(np.uint8))
        
    eq_img_array = np.zeros(img_array.shape)
    
    for i in range(Wi):
        for j in range(Hi):
            eq_img_array[i][j] = tr_map[img_array[i][j]]
    
    return eq_img_array


s = sys.argv[1]
img = cv2.imread(s)
k = get_index(s)

if k==1:
    fimg1 = gamma_transform(gray_patch(img,4),0.5)
    cv2.imwrite('enhanced-cctv1.jpg', fimg1)

elif k==2:
    fimg2 = gamma_transform(img, 0.5)
    cv2.imwrite('enhanced-cctv2.jpg', fimg2)

elif k==3:
    fimg3 = histogram_equalization(img)
    cv2.imwrite('enhanced-cctv3.jpg', fimg3)

elif k==4:
    fimg4 = histogram_equalization(img)
    cv2.imwrite('enhanced-cctv4.jpg', fimg4)