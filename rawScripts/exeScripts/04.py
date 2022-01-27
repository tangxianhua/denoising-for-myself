import numpy as np
import pandas as pd
import os
#cal-snr
# print('calculate-snr')
# print("请输入(空格隔开)文件1绝对路径 文件2绝对路径 csv文件存储绝对路径 图像width 图像height 图像位数(8或16) 计算窗口左上角y坐标 右下角y坐标 左上角x坐标 左下角x坐标")
# totalinput = input()
# totalinput = totalinput.split(' ')
# if len(totalinput)!= 10:
#     print(r'参数错误! 命令格式为：input1path input2path csvoutputpath width height imagebit y1 y2 x3 x4')
# else:
#     path1 = totalinput[0]
#     path2 = totalinput[1]
#     savepath = totalinput[2]
#     height = int(totalinput[3])
#     width = int(totalinput[4])
#     imagebit = int(totalinput[5])
#     win = [int(n) for n in totalinput[6:]]
#     list1 = []
#     if imagebit == 8:
#         npy1 = np.fromfile(path1, dtype='uint8')
#         npy1 = npy1.reshape(width, height)
#         npy1 = np.array(npy1)
#         npy2 = np.fromfile(path2, dtype='uint8')
#         npy2 = npy2.reshape(width, height)
#         npy2 = np.array(npy2)
#
#         Region1 = npy1[win[0]:win[1],win[2]:win[3]]
#         Region2 = npy2[win[0]:win[1],win[2]:win[3]]
#         S = np.linalg.norm(Region1)
#         MSE = np.linalg.norm(Region1-np.mean(Region2))
#         SN = 20*(np.log10(S/MSE))
#         list1.append(SN)
#         listall = list1
#         listcsv = pd.DataFrame(data=listall,index=['SN'])
#         savename = os.path.join(savepath,'result.csv')
#         listcsv.to_csv(savename)
#     if imagebit == 16:
#         npy1 = np.fromfile(path1, dtype='uint16')
#         npy1 = npy1.reshape(width, height)
#         npy1 = np.array(npy1)
#         npy2 = np.fromfile(path2, dtype='uint16')
#         npy2 = npy2.reshape(width, height)
#         npy2 = np.array(npy2)
#         Region1 = npy1[win[0]:win[1], win[2]:win[3]]
#         Region2 = npy2[win[0]:win[1], win[2]:win[3]]
#         S = np.linalg.norm(Region1)
#         MSE = np.linalg.norm(Region1 - np.mean(Region2))
#         SN = 20 * (np.log10(S / MSE))
#         list1.append(SN)
#         listall = list1
#         listcsv = pd.DataFrame(data=listall,index=['SN'])
#         savename = os.path.join(savepath, 'result.csv')
#         listcsv.to_csv(savename)
#


# import glob
# import cv2
# from matplotlib import pylab as plb
# path1 = r'C:\Users\xianhua.tang\Desktop\denoiseTest\bilateral\iso6400\sfr\samegain\sony6400\plottest\rgboutjpg'
# npy1list = glob.glob(os.path.join(path1,'*.jpg'))
# npy2list = glob.glob(os.path.join(path1,'*.jpg'))
# npy1list.sort()
# npy2list.sort()
# # width = 4024
# # height = 6048
# list1 = []
# list2 = []
# savepath = r'C:\Users\xianhua.tang\Desktop\denoiseTest\bilateral\iso6400\sfr\samegain\sony6400\plottest'
# for i in range(len(npy1list)):
#     print(i)
#     list2.append(str(npy1list[i]))
#     # npy1 = np.fromfile(npy1list[i], dtype='uint16')
#     # npy1 = npy1.reshape(width, height)
#     npy1 = cv2.imread(npy1list[i])[:,:,1]
#     npy1 = np.array(npy1)
#     # npy2 = np.fromfile(npy2list[i], dtype='uint16')
#     # npy2 = npy2.reshape(width, height)
#     npy2 = cv2.imread(npy2list[i])[:,:,1]
#     npy2 = np.array(npy2)
#
#     Region1 = npy1[1355:2608, 1954:4023]
#     Region2 = npy2[1355:2608, 1954:4023]
#     plb.imshow(Region1,'gray')
#     plb.show()
#     S = np.linalg.norm(Region1)
#     MSE = np.linalg.norm(Region1 - np.mean(Region2))
#     SN = 20 * (np.log10(S / MSE))
#     list1.append(SN)
#     listall = list1
#     listcsv = pd.DataFrame(data=listall,index=list2)
#     savename = os.path.join(savepath, 'result.csv')
#     listcsv.to_csv(savename)


# import glob
import cv2
from matplotlib import pylab as plb
imagepath = r'C:\Users\xianhua.tang\Desktop\denoiseTest\bilateral\iso6400\sfr\samegain\sony6400\plottest\rgboutjpg\j1g1t3-j01-g01-RGB_out2021-12-13-15-17-10-014.jpg'
image = cv2.imread(imagepath)[:,:,1][1275:1385,2309:2416]#[1513:1629,2918:3087]#[1325:2604,1984:3985]#[391:817,2621:3242]
# image = np.fromfile(r'D:\Project\Python\Data\Data\sources\temp\DSC00133.raw',dtype = 'uint16')
# height=6048
# width=4024
# imgData = image.reshape(width, height)
# img = imgData[358:997,2761:3384]
print(image.shape)
# plb.imshow(image,'gray')
# plb.show()
S = np.linalg.norm(image)
MSE = np.linalg.norm(image - np.mean(image))
print(S)
print(np.mean(image))
print(MSE)
SN = 20 * (np.log10(S / MSE))
print(SN)