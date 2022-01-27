import numpy as np
import cv2
from matplotlib import pylab as plb

# height=6048
# width=4024
# channel = 3
# img=np.fromfile(r'D:\Project\Python\Data\Data\sources\bm3d\gamma\ISO100\DS_out-1915.raw',dtype = 'uint16')
# imgData = img.reshape(width, height,channel)#4024,6048
# img = np.array(imgData)[:,:,2]
# newimage = np.array(img[1500:2500,2800:3800])
# # print(img.shape)
# # print(img.max())
# # img2 = img[:,:,1]
# # #img = np.load(path)
# # print(img2.max())#最大值
# print(np.mean(newimage))#均值
# # print(np.median(img2))#中值
# print(newimage.std())#标准差
# # print(img2.var())#方差
# # plb.imshow(newimage,'gray')
# # plb.show()

img = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin\DSC02030-g2.npy')
img = img[690:1280,1100:1900]
print(np.mean(img))#均值
print(img.std())
print(img.var())
# plb.imshow(img,'gray')
# plb.show()