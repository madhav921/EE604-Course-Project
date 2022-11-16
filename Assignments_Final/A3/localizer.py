import cv2
import sys

def f1(img):
    res = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res += abs(int(img[i][j][1])-int(img[i][j][2]))
    return abs(res)/(img.shape[0]*img.shape[1])


def f2(img):
    res = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res += abs(2*int(img[i][j][1])-int(img[i][j][0])-int(img[i][j][2]))
    return abs(res)/(img.shape[0]*img.shape[1])

def swap02(img):
    red = img[:,:,2].copy()
    blue = img[:,:,0].copy()
    img[:,:,0] = red
    img[:,:,2] = blue
    return img


s = sys.argv[1]
img = cv2.imread(s)
img = swap02(img)

loc = img
a = f1(loc)
b = f2(loc)
# print(a,b)
if (a>=0 and a<=6) and (b>=1 and b<=8):
    print("1")
elif (a>=0 and a<=80) and (b>=12 and b<=85):
    print("2")
elif (a>6 and a<=20) and (b>=0 and b<=12):
    print("3")
else:
    print("2")

# building=1, grass=2 and road=3
