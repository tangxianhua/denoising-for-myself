import cv2  #OpenCV包
import numpy as np
import rawpy
from matplotlib import pylab as plb
from PIL import Image
# import Image


# height=4024
# width=6048
# #channels =1
# img=np.fromfile(r'D:\Project\Python\Data\Data\sources\temp\WB_out.raw',dtype = 'uint16')
# imgData = img.reshape(width, height)
#
# countR =0
# countnum = 0
# for i in range(0,6048,2):
#     for j in range(0,4024,2):
#            a =np.fromiter(imgData[i][j], dtype=int)
#            countR=countR+int(a)
#            countnum+=1
# countB =0
# for i in range(1,6048,2):
#     for j in range(1,4024,2):
#            a = np.fromiter(imgData[i][j], dtype=int)
#            countB += int(a)
# countG1 =0
# for i in range(1,6048,2):
#     for j in range(0,4024,2):
#            a = np.fromiter(imgData[i][j], dtype=int)
#            countG1 += int(a)
# countG2 =0
# for i in range(0,6048,2):
#     for j in range(1,4024,2):
#            a = np.fromiter(imgData[i][j], dtype=int)
#            countG2 += int(a)
#
# countR = countR/6084288
# countB = countB/6084288
# countG1 = countG1/6084288
# countG2 = countG2/6084288
#
# print(countR)
# print(countB)
# print(countG1)
# print(countG2)


img = np.load(r'D:\Project\Python\Data\Data\sources\temp\BUGTEST\45\16bit\WB_out.npy')#[1200:2400,2400:3600]
print(img.shape)

width = 6048
height = 4024
countnum = 0
countR =0
for i in range(0,height,2):
    for j in range(0,width,2):
           a = img[i][j]
           countR=countR+int(a)
           countnum+=1

countB =0
for i in range(1,height,2):
    for j in range(1,width,2):
           a = img[i][j]
           countB += int(a)

countG1 =0
for i in range(0,height,2):
    for j in range(1,width,2):
           a = img[i][j]
           countG1 += int(a)
countG2 =0
for i in range(1,height,2):
    for j in range(0,width,2):
           a = img[i][j]
           countG2 += int(a)

totalnumber = 6084288
countR = countR/totalnumber
countB = countB/totalnumber
countG1 = countG1/totalnumber
countG2 = countG2/totalnumber

print(countR)
print(countB)
print(countG1)
print(countG2)
print(countnum)