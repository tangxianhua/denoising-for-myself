import cv2
from matplotlib import pylab as plb
import numpy as np
#读raw图测试
# import cv2
# import numpy as np
# # 首先确定原图片的基本信息：数据格式，行数列数，通道数
# rows=1080#图像的行数
# cols=1920#图像的列数
# channels =1# 图像的通道数，灰度图为1
# # 利用numpy的fromfile函数读取raw文件，并指定数据格式
# img=np.fromfile(r'D:\Software\PycharmCommunity2018\Data\Circle200.raw', dtype='uint8')
# print(img.shape)
# # 利用numpy中array的reshape函数将读取到的数据进行重新排列。
# img=img.reshape(rows, cols, channels)
# # 展示图像
# cv2.imshow('Infared image-640*512-8bit',img)
# # 如果是uint16的数据请先转成uint8。不然的话，显示会出现问题。
# cv2.waitKey()
# cv2.destroyAllWindows()
# print('ok')


#npy保存
#path = r"D:\Project\Python\Data\Data\sources\bm3d\DSC01831-gaus500.npy"
# path2 = r"D:\Project\Python\Data\nlmeans3-10.npy"
#out = np.load(path)
# print(out[3,4])
# out2 = np.load(path2)
# out = out.astype(np.uint16)
# out2 = out2.astype(np.uint16)
# # cv2.imwrite('nlmeans.tif', out)
# # cv2.imwrite('nlmeans3-10.tif', out2)
# plb.imshow(out,'gray')
# plb.show()
#save = out.astype(np.uint16)
#cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\DSC01831-gaus500.tif', save)


#凯泽窗
# import numpy as np
# from matplotlib.pyplot import plot, show
# # (1) 调用kaiser函数，以计算凯泽窗：
# window = np.kaiser(42, 14)
# # (2) 使用Matplotlib绘制凯泽窗：
# plot(window)
# show()

#贝塞尔函数
# import numpy as np
# from matplotlib.pyplot import plot, show
# # (1) 使用NumPy的linspace函数生成一组均匀分布的数值。
# x = np.linspace(0, 4, 100)
# # (2) 调用i0函数进行计算：
# vals = np.i0(x)
# # (3) 使用Matplotlib绘制修正的贝塞尔函数：
# plot(x, vals)
# show()

#裁剪噪声
# path2 = r'D:\Project\Python\Data\Data\sources\bm3d\gamma\ISO100\800\1916-B.npy'
# # npy1 = np.load(path1)
# # npy2 = np.load(path2)
# # npycut = npy1-npy2
# #cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\npycut.tif', npycut)
# plb.imshow(npycut,'gray')
# plb.show()






#三通道保存
# height=400
# width=400
# channel = 1
# path1 = r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\BIL\1917-B-bil.npy'
# path2 = r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\BIL\1917-G-bil.npy'
# path3 = r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\BIL\1917-R-bil.npy'
# img1 = np.load(path1)
# img2 = np.load(path2)
# img3 = np.load(path3)
# print(img1.shape)
# imgData1 = img1.reshape(width, height,channel)
# imgData2 = img2.reshape(width, height,channel)
# imgData3 = img3.reshape(width, height,channel)
# imgnew = np.concatenate((imgData1,imgData2,imgData3),axis=2)
# imgnew = imgnew.astype(np.uint16)
# cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\BIL\1917-rgb.tif',imgnew)
# plb.imshow(imgnew,'gray')
# plb.show()
# image = cv2.imread(r'D:\Project\Python\Data\Data\sources\bm3d\gamma\1918and1917full\DSC01917.JPG')
# print(image.shape)


# height=6048
# width=4024
# channel = 3
# img=np.fromfile(r'D:\Project\Python\Data\Data\sources\bm3d\gamma\GM_out.raw',dtype = 'uint16')
# print(img.shape)
# imgData = img.reshape(width, height,channel)#4024,6048
# R = imgData[:,:,2]
# print(R.shape)
# # image = np.array(imgData,dtype = 'uint16')
# newimage = np.array(R[1000:1500,3000:3500])
# # # print(newimage.shape)
# # # newimage = newimage.flatten()
# # # print(max(newimage))
# # # noise = np.random.normal(0, 0.2 ** 0.5, newimage.shape)
# # # newimage = newimage + noise
# # #cv2.imwrite(r'D:\Project\Python\Data\Data\sources\others\DSC01831-guas500.jpg',newimage)
# plb.imshow(newimage,'gray')
# plb.show()
# np.save(r'D:\Project\Python\Data\Data\sources\bm3d\gamma\GM_out-500B.npy',newimage)


#去水印
# path = r'D:\Project\Python\Data\Data\sources\temp\wang.png'
# img = cv2.imread(path)
# img = np.array(img)
# #new = np.zeros((2608, 1654))
#
# #print(img.shape)#2608,1654  250,235 232,231,236
# for i in range(2608):
#     for j in range(1654):
#         if (img[i][j][0] >= 230) &(img[i][j][0] <= 250):
#             img[i][j][0] = 255
#             img[i][j][1] =255
#             img[i][j][2] =255
# # np.save(r'D:\Project\Python\Data\Data\sources\temp\wang2.npy',img)
# # #cv2.imwrite(r'D:\Project\Python\Data\Data\sources\temp\wang2.psd',img)
# # plb.imshow(img,'gray')
# # plb.show()


#ATER-DEMOSIC转三通道
# height=6048
# width=4024
# channel = 3
# img=np.fromfile(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\DS_out1917.raw',dtype = 'uint16')
# print(img.shape)
# imgData = img.reshape(width, height,channel)
# print(imgData.shape)
# imgr = imgData[:,:,2]
# np.save(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\1917-R.npy',imgr)
# # plb.imshow(ce,'gray')
# # plb.show()

# img = cv2.imread(r'D:\Project\Python\Data\Data\sources\bm3d\gamma\1918\DSC01917.JPG')
# img = np.array(img[2800:3600,2200:3000])
#
# img = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\1917-R.npy')
# newimage = np.array(img[2800:3200,2600:3000])
# np.save(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\1917-R.npy',newimage)
# cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\1917-R.tif', newimage)
# plb.imshow(newimage,'gray')
# plb.show()

#查看npy图
# img = np.load(r'D:\Project\Python\Data\Data\sources\npy\DSC00017-g2.npy')
# img.tofile(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\DSC00017-g2.raw')
# plb.imshow(img,'gray')
# plb.show()
# height = 2
# width = 4
# list1 = [[1,2,3,4],[5,6,7,8]]
# arr = np.array(list1)

# arr = arr.reshape(height,width )
# print((arr)**2)
#img = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin\DSC02030.npy')[1238:2538,2073:3573]
#img = np.load(r'D:\Project\Python\Data\Data\sources\temp\npy\WB_out.npy')#[1200:2400,1200:2400]
#print(img.shape)
# img = img[:,:,2]
# np.save(r'D:\Project\Python\Data\Data\sources\temp\npy\RGB_out-b.npy',img)
# img.tofile(r'D:\Project\Python\Data\Data\sources\temp\raw\RGB_out-b.raw')
# plb.imshow(img,'gray')
# plb.show

# from matplotlib import pyplot as plt
#img = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin\DSC02030.npy')
# img = cv2.imread(r'D:\Project\Python\Data\Data\sources\temp\02.png')
# plb.imshow(img)
# plb.show()
# myslice = slice(0,-1,None)
# #list1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# list1 = [[[0,2],[3,4],[7,5]],
#          [[1,3],[4,5],[8,6]]]
# #
# list1 = np.array(list1)
# temp
# print(list1)
# print(list1.shape)
# a = list1.sum(0)
# b = list1.sum(axis=0)
# #print(list1[myslice])
# print(a)
# print(b)

# image = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\xilinx\origin\U0002.npy')[:,:,2]
# image = image.astype(np.uint16)
# np.save(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\xilinx\origin\U0002-b.npy',image)


import os
path= r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\xilinx\jgt\tvm'
imagename = 'jbilguidetvm-01-'
rimage = np.load(os.path.join(path,imagename+'r.npy'))
# print(rimage.shape)
# plb.imshow(rimage,'gray')
# plb.show()
gimage = np.load(os.path.join(path,imagename+'g.npy'))
bimage = np.load(os.path.join(path,imagename+'b.npy'))
newimage = np.zeros((2160,3840,3))

for i in range(2160):
    for j in range(3840):
        newimage[i][j][0] = rimage[i][j]
        newimage[i][j][1] = gimage[i][j]
        newimage[i][j][2] = bimage[i][j]
newimage = newimage.astype(np.uint16)
newimage.tofile(os.path.join(path,imagename+'.raw'))