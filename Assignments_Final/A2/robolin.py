import numpy as np
from skimage import io
from skimage.color import rgb2gray
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

def conv(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    kernel = np.flip(kernel)
    for i in range(Hi):
      for j in range(Wi):
        multi = padded[i:i+Hk , j:j+Wk]
        out[i][j]=(multi*kernel).sum()
    return out


def partial_x(img):
    out = None
    kernel = np.array([[1,0,-1]])
    out = conv(img,kernel)
    out = out/2
    return out

def partial_y(img):
    out = None
    kernel = np.array([[1],[0],[-1]])
    out = conv(img,kernel)
    out = out/2
    return out



def partial_x1(img):
    out = None
    kernel = np.array([[-1,0,1]])
    out = conv(img,kernel)
    out = out/2
    return out

def partial_y1(img):
    out = None
    kernel = np.array([[-1],[0],[1]])
    out = conv(img,kernel)
    out = out/2
    return out


def gradient(img):
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    H,W = img.shape

    G_x = partial_x(img)
    G_y = partial_y(img)
    G = np.sqrt(G_x**2+G_y**2)
    for i in range(H):
      for j in range(W):
        if G_x[i][j]==0:
          theta[i][j]=np.pi/2
        else :
          theta[i][j]=np.arctan(G_y[i][j]/G_x[i][j])
        if theta[i][j]<0:
          if G_x[i][j]<0:
            theta[i][j]+=2*np.pi;
          else :
            theta[i][j]+=np.pi;
        else :
          if G_x[i][j]<0 or G_y[i][j]<0:
            theta[i][j]+=np.pi
    theta = theta/np.pi*180
    return G, theta


def non_maximum_suppression(G, theta):
    H, W = G.shape
    out = np.zeros((H, W))
    a = np.zeros((H+2,W+2))
    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    for i in range(H):
      for j in range(W):
        a[i+1][j+1]=G[i][j]
    
    for i in range(1,H+1):
      for j in range(1,W+1):
        if theta[i-1][j-1]==45 or theta[i-1][j-1]==225:
          if a[i][j]<a[i+1][j+1] or a[i][j]<a[i-1][j-1]:
            out[i-1][j-1]=0
          else :
            out[i-1][j-1]=a[i][j]
        elif theta[i-1][j-1]==90 or theta[i-1][j-1]==270:
          if a[i][j]<a[i+1][j] or a[i][j]<a[i-1][j]:
            out[i-1][j-1]=0
          else :
            out[i-1][j-1]=a[i][j]
        elif theta[i-1][j-1]==135 or theta[i-1][j-1]==315:
          if a[i][j]<a[i-1][j+1] or a[i][j]<a[i+1][j-1]:
            out[i-1][j-1]=0
          else :
            out[i-1][j-1]=a[i][j]
        elif theta[i-1][j-1]==0 or theta[i-1][j-1]==180 or theta[i-1][j-1]==360:
          if a[i][j]<a[i][j+1] or a[i][j]<a[i][j-1]:
            out[i-1][j-1]=0
          else :
            out[i-1][j-1]=a[i][j]
    return out


def double_thresholding(img, high, low):
    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)
    (H, W) = img.shape
    for i in range(H):
      for j in range(W):
        if img[i][j]>high:
          strong_edges[i][j]=img[i][j]
        elif img[i][j]>low :
          weak_edges[i][j]=img[i][j]
    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    neighbors = []
    t=0
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))
                t+=1
    return neighbors,t


def link_edges(strong_edges, weak_edges):
    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    Hi, Wi = indices.shape
    while 1:
      indices = np.stack(np.nonzero(edges)).T
      Hi,Wi=indices.shape
      m=0
      for x in range(Hi):
        t,a=get_neighbors(indices[x][0],indices[x][1],H,W)
        for y in range(a):
          if edges[t[y][0]][t[y][1]]==0 and weak_edges[t[y][0]][t[y][1]]!=0:
            m=1
            edges[t[y][0]][t[y][1]]=weak_edges[t[y][0]][t[y][1]]
      if m==0:
        break
    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    Hi, Wi = img.shape
    edge = np.zeros((Hi, Wi))
    kernel = gaussian_kernel(kernel_size,sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    return edge


def hough_image(img,edge):
    dst = np.uint8(edge)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 120, None, 0, 0)
    if lines is not None:
        for i in range(0, int(len(lines)/2),2):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
        for i in range(int(len(lines)/2), len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    return img
    
    
s = sys.argv[1]
img = io.imread(s)
k = get_index(s)
if s[0]=='.':
    s = s[2:]

img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = canny(img_gray, kernel_size=5, sigma=1.5, high=0.03, low=0.02)
final = hough_image(img,edges)
final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
# red = final[:,:,2].copy()
# blue = final[:,:,0].copy()

# final[:,:,0] = red
# final[:,:,2] = blue

cv2.imwrite("robolin-"+s, final)