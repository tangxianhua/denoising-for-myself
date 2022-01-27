
import math
import numpy as np
# sigma = 1
# temp = 1/(2.5066282746)#1/根号(2Π*sigma**2)
# print(temp)
# print(math.exp(-1))
# print(math.exp(-0.5))
# print(temp*math.exp(-1))
# print(temp*math.exp(-0.5))

#计算高斯权重
# num = 2*3.141592653589793
# num_sqrt = num ** 0.5
#print(' %0.10f 的平方根为 %0.10f'%(num ,num_sqrt))
#temp = 0.36787944+0.60653066+0.36787944+0.60653066+1+0.60653066+0.36787944+0.60653066+0.36787944
# temp = 0.99918401+0.99959192+0.99918401+0.99959192+1+0.99959192+0.99918401+0.99959192+0.99918401
# print(0.99918401/temp)
# print(0.99959192/temp)
# print(1/temp)
def getClosenessWeight(sigma_g,H,W):
    r,c = np.mgrid[0:H:1,0:W:1]
    r=r.astype(np.float64)
    c=c.astype(np.float64)
    r-=(H-1)/2
    c-=(W-1)/2
    closeWeight = np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigma_g,2))
    return closeWeight
closenessWeightr = getClosenessWeight(35,3,3)#高斯
list35 = []
print(closenessWeightr)
for i in range(3):
     for j in range(3):
          list35.append(closenessWeightr[i][j])
print(list35)

#高斯权重扩大查找表
# import numpy as np
# listr = [0.99750312, 0.99875078, 0.99750312,0.99875078, 1., 0.99875078,0.99750312, 0.99875078, 0.99750312]#8.9850156
# listb = [0.99840128, 0.99920032, 0.99840128,0.99920032, 1., 0.99920032,0.99840128, 0.99920032, 0.99840128]#8.9904064
# listg = [0.36787944, 0.60653066, 0.36787944,    0.60653066, 1.,  0.60653066,     0.36787944, 0.60653066, 0.36787944]
# listr = np.asarray(listr)
# listb = np.asarray(listb)
# listg = np.asarray(listg)
# count = 0
# for i in range(len(listg)):
#      listg[i]= listg[i]*1048576
# print(listg)

#jbil权重查找表
# import math
# import matplotlib.pyplot as plt
# import os
# listy = []
# sigmar = 50
# sigmab = 50
# sigmag = 50
# times = 1024
# savepathname = os.path.join(r'D:\Project\Done\DenoiseBatch\test','jbilLUT_'+str(sigmar)+'_'+str(sigmab)+'_'+str(sigmag)+'.txt')
# #r的查找表
# for i in range(0,1024):
#         temp = ((i*i)/(sigmar*sigmar))*-0.5
#         num = math.exp(temp)#exp(abs(f(i,j)-f(k,l))**2*-0.5)
#         num = num * times
#         if num >= 1:
#         #     listy.append(int((num*1024)))
#         #     listx.append(i)
#           listy.append(int(num))
#         else:
#           listy.append(1)
# lengthr = len(listy)
# print('lenr',len(listy))
# #b的查找表
# for i in range(0,1024):
#         temp = ((i*i)/(sigmab*sigmab))*-0.5
#         num = math.exp(temp)#exp(abs(f(i,j)-f(k,l))**2*-0.5)
#         num = num * times
#         if num >= 1:
#           listy.append(int(num))
#         else:
#           listy.append(1)
#
# lengthb = len(listy)-lengthr
# print('b',lengthb)
# print('lenrb',len(listy))
# #g的查找表
# for i in range(0,1024):
#         temp = ((i*i)/(sigmag*sigmag))*-0.5
#         num = math.exp(temp)#exp(abs(f(i,j)-f(k,l))**2*-0.5)
#         num = num * times
#         if num >= 1:
#           listy.append(int(num))
#         else:
#           listy.append(1)
# lengthg = len(listy)-lengthb-lengthr
# print('g',lengthg)
# print('lenrbg',len(listy))
# file = open(savepathname,'w')
# lenth = len(listy)
# for i in range(lenth):
#     file.write(str(int(listy[i])))
#     file.write('\n')
# file.close()
# print()
# plt.plot(listy)
# plt.show()
# print(listy)
# print(len(listx))


#散点图
# import matplotlib.pyplot as plt
# listy = [  	0.2149,	0.2124,	0.2131,	0.2135,	0.2134,	0.2143,	0.2023,	0.2006,	0.2002,	0.2151,	0.2136,	0.2107,	0.2142,	0.2141,	0.2129,	0.1998,	0.1994,	0.1987,	0.2138,	0.2121,	0.213,	0.214,	0.2139,	0.2133,	0.1983,	0.1992,	0.1976,	0.2148,	0.2121,	0.2138,	0.2132,	0.2133,	0.2141,	0.199,	0.1989,	0.1979]
# listx = [		6.422615906,	6.424236296,	6.425293292,	6.434736465,	6.436346994,	6.437316227,	6.498451159,	6.499661801,	6.500350022,	6.427164803,	6.428785367,	6.429806403,	6.438553579,	6.440071886,	6.441030392,	6.500535364,	6.501608758,	6.502296042,	6.430906858,	6.432560044,	6.43359439,	6.441938825,	6.443320755,	6.444178581,	6.502366668,	6.503411419,	6.504124794,	6.42747535,	6.429147989,	6.429892403,	6.438680556,	6.440241849,	6.441155777,	6.500609179,	6.501735731,	6.502403043]
# plt.title('SNR-MTF50',fontsize=24)
# plt.xlabel('SNR(8bit-jpg-Gchannel)',fontsize=14)
# plt.ylabel('MTF50(jpg)',fontsize=14)
# plt.scatter(listx,listy)
# plt.show()

# #差值对比
# import numpy as np
# from matplotlib import pylab as plb
# import cv2
# from PIL import Image
# npyc = np.load(r'D:\Project\Python\Data\Data\sources\temp\2029\tvm\int\cefoloat.npy')[500:3900,100:6000]
# npyc = npyc.astype(np.float64)
# print(npyc.shape)
# npypython = np.load(r'D:\Project\Python\Data\Data\sources\temp\2029\tvm\int\ceaug.npy')[500:3900,100:6000]
# npypython = npypython.astype(np.float64)
# print(npypython.shape)
# diff = abs(npyc-npypython)
# count = 0
# for i in range(diff.shape[0]):
#     for j in range(diff.shape[1]):
#         if (diff[i][j]>100):
#             # print(diff[i][j])
#             # print('i',i)
#             # print('j',j)
#             count+=1
# print('差值最大值',diff.max())
# print(diff)
# print('差值大于100个数count',count)
# # #newimage = np.resize(npyc,(1600,1600))
# # dst = cv2.resize(npyc,(100,100),interpolation = cv2.INTER_LINEAR)
# # # dst2 = cv2.resize(dst,(800,800),interpolation = cv2.INTER_LINEAR)
# plb.imshow(diff,'gray')
# plb.show()
# # dst2.tofile(r'D:\Project\Python\Data\Data\sources\temp\opencv800.raw')

#resize测试
# image = Image.open(r'D:\Project\Python\Data\Data\sources\temp\DSC02029(1).jpg')
# pilresize = image.resize((400, 400),Image.ANTIALIAS)
# plb.imshow(pilresize,'gray')
# plb.show()

#tvm查找表
# import cmath
# import pandas as pd
# # print('{0} 的平方根为 {1:0.3f}+{2:0.3f}j'.format(num, num_sqrt.real, num_sqrt.imag))
# import numpy as np
# import matplotlib.pyplot as plt
# listx = []
# listy = []
# count = 0
# total = 100000
# for i in range(0,total):
#         num_sqrt = cmath.sqrt(i)
#         listy.append(num_sqrt.real)
#         listx.append(count)
#         # count+=1
# # plt.plot(listy)
# # plt.show()
# listy = np.asarray(listy)
# # plt.scatter(listx,listy)
# # plt.show()
# file = open(r'D:\Project\Python\Data\Data\sources\temp\cee-100000-16times.txt','w')
# lenth = len(listy)
# for i in range(lenth):
#     file.write(str(int(listy[i])))
#     file.write('\n')
# file.close()


#统计元素个数
# import numpy as np
# from  matplotlib import pylab as plb
# import matplotlib.pyplot as plt
# from collections import  Counter
# from tqdm import trange
# # # image = np.load(r'D:\Project\Python\Data\Data\sources\temp\6946\huijie\origin\14bit\lsb5-14bit.npy')
# # # print(image.shape)
# # # M = plt.imread(r'D:\Project\Python\Data\Data\sources\temp\6946\huijie\denoise\jbil\5.bmp')[:,:,1]
# npy = np.load(r'D:\Project\Python\Data\Data\sources\temp\tvmtempr.npy')
# npy = npy.astype(np.uint64)
# # npy = npy.flatten()
# # Counter(npy)  # {label:sum(label)}
# # list1 =[]
# # for i in trange(8550041579):
# #   temp = sum(npy==i)
# #   list1.append(temp)
# # # plb.imshow(npy,'gray')
# # # plb.show()
# from collections import Counter
# #print(Counter(npy.flatten()))
# list1 = list(Counter(npy.flatten()).values())
# plt.plot(list1)
# plt.show()

# import numpy as np
# from  matplotlib import pylab as plb
# width =4024
# height = 6048
# path = r'D:\Project\Done\Denoise\test\resultcompare\snr\rgb\RGB_j01g01t01.raw'
# npy1 = np.fromfile(path, dtype='uint8')
# npy1 = npy1.reshape(width, height,3)
# npy1 = np.array(npy1)[:,:,1]#[1607:2172,1538:2120]
# #print(np.mean(npy1))
# plb.imshow(npy1,'gray')
# plb.show()

import numpy as np
# list1 = [1,100,101,201,203,205,207,301,303,
#          305,307,309,311,313,315,401,403,405,
#          407,409,411,413,415,417,419,421,423,
#          425,427,429,431]
#
# #print(415 in list1)
# file = open(r'D:\Project\Done\Resize0715\refer\bicubicweights1.txt','w')
# for i in range(0,432):
#     if i in list1:
#         file.write(str(i))
#         file.write('* ')
#         file.write('* ')
#         file.write('* ')
#         file.write('*')
#         file.write('\n')
#     else:
#         file.write(str(0)+' ')
#         file.write(str(0)+' ')
#         file.write(str(0)+' ')
#         file.write(str(0))
#         file.write('\n')
# file.close()
import numpy as np
from matplotlib import pylab as plb
img=np.fromfile(r'D:\Project\Python\Data\Data\sources\temp\RGB.raw',dtype = 'uint8')
#imgData = img.reshape(width, height,channel)
image = img.reshape(4024, 6048)[1500:2300,2500:3300]
image.tofile(r'D:\Project\Done\Resize0120\Resize\ref\RGB800.raw')
#image = imgData[1607:2172,1538:2120]
# plb.imshow(image,'gray')
# plb.show()