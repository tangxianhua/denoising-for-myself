import numpy as np
import pandas as pd
import os
import math
import cv2
import matplotlib.pyplot as plt
#mse计算
def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse
def sn(S,MSE):
    SN = 20*(np.log10(S/MSE))
    return SN
def PSNR(maxvalue,mse):
    return 20 * np.log10(maxvalue / mse)


#单张测试
# list6946rgb = [[135, 298, 177, 338],[195, 299, 238, 342],[251,269,296,312],[80, 264, 121, 306],[284,211,325,257],
# [56, 206, 93, 249],[286,155,327,199],[59, 149, 94, 192],[257,99,297,140],[88, 96, 125, 136],[199,69,241,107],[143,70,182,107]]
# list240rgb = [[543,764,1163,1308],[690,1656,1245,2178],[848,2482,1375,2934],[989,3200,1489,3581],
#               [1435,677,2077,1281],[1528,1623,2131,2172],[1631,2472,2191,2939],[1734,3206,2256,3608],
#               [2381,677,2990,1270],[2436,1601,3012,2156],[2468,2444,3012,2907],[2506,3195,3028,3554],
#               [3349,666,3920,1270],[3338,1612,3866,2113],[3317,2466,3817,2879],[3311,3195,3779,3521],
#               [4214,677,4714,1264],[4192,1580,4644,2085],[4132,2401,4557,2820],[4062,3124,4470,3472],
#               [5041,715,5438,1270],[4964,1580,5345,2026],[4888,2363,5231,2754],[4779,3059,5111,3380]]
# listNoise = list240rgb
# chartnum = 24
# #多张图
# # listbmp = list6946rgb
# # listmy = list6946rgb
# csvsavepath = r'D:\Project\Done\Denoise\test\resultcompare'
# savename = 'snr-method3-020303-24chart'
# maxvalue = 16383
# #npy
# #noise =np.load(r'D:\Project\Python\Data\Data\sources\temp\6946\huijie\origin\14bit\RGB_lsb5-14.npy')[:,:,1]
# #jpg
# noise = cv2.imread(r'D:\Project\Done\Denoise\test\resultcompare\snr\method3\jpg\020303.JPG')
# #多张图
# # my = np.load(r'D:\Project\Python\Data\Data\sources\temp\6946\huijie\denoise\jbil\RGB_jbillsb5-101520.npy')[:,:,1]
# # bmp = plt.imread(r'D:\Project\Python\Data\Data\sources\temp\6946\huijie\denoise\jbil\5.bmp')[:,:,1]
# SNlistnoise = []
# # SNlistmy = []
# # SNlistbmp = []
# for i in range(chartnum):
#     chartRegionNoise = noise[listNoise[i][1]:listNoise[i][3],listNoise[i][0]:listNoise[i][2]]
#     #print(np.mean(chartRegionNoise))
#     S = np.linalg.norm(chartRegionNoise)
#     msesn = np.linalg.norm(chartRegionNoise - np.mean(chartRegionNoise))
#     snresult = sn(S,msesn)
#     SNlistnoise.append(snresult)
#     # chartRegionmy = my[listmy[i][1]:listmy[i][3],listmy[i][0]:listmy[i][2]]
#     # S = np.linalg.norm(chartRegionmy)
#     # msesn = np.linalg.norm(chartRegionmy - np.mean(chartRegionmy))
#     # snresult = sn(S,msesn)
#     # SNlistmy.append(snresult)
#     # chartRegionbmp = bmp[listbmp[i][1]:listbmp[i][3],listbmp[i][0]:listbmp[i][2]]
#     # S = np.linalg.norm(chartRegionbmp)
#     # msesn = np.linalg.norm(chartRegionbmp - np.mean(chartRegionbmp))
#     # snresult = sn(S,msesn)
#     # SNlistbmp.append(snresult)
# listall = [SNlistnoise]
# listcsv = pd.DataFrame(data=listall,index = ['SNR'])
# print('保存路径',os.path.join(csvsavepath,savename+'.csv'))
# listcsv.to_csv(os.path.join(csvsavepath,savename+'.csv'))


#rawrgb,jpg文件夹批处理
import glob
import os
import cv2
path = r'D:\Project\Done\DenoiseBatch\test\result\snr\20220126\rgb'
csvsavepath = r'D:\Project\Done\DenoiseBatch\test\result\snr\20220126\rgb'
savename = 'snr'
imagelist = glob.glob(os.path.join(path,'*.jpg'))
imagelist.sort()
width = 4024
height = 6048
SNlistnoise = []
listname = []
for i in range(len(imagelist)):
    print(imagelist[i])
    basename = (os.path.basename(imagelist[i])).split('.')[0]
    listname.append(str(basename))
    #print(listname)
      #rawrgb
#     image = np.fromfile(imagelist[i], dtype='uint8')
#     image = image.reshape(width, height, 3)
#     image = np.array(image)[:, :, 1][1607:2172, 1538:2120]
      #jpg
    image = cv2.imread(imagelist[i])[:, :, 1][1607:2172, 1538:2120]
    S = np.linalg.norm(image)
    msesn = np.linalg.norm(image - np.mean(image))
    snresult = sn(S, msesn)
    SNlistnoise.append(snresult)
    #print(snresult)
listall = [SNlistnoise]
listcsv = pd.DataFrame(data=listall,index = ['SNR'],columns=listname)
print('保存路径',os.path.join(csvsavepath,savename+'.csv'))
listcsv.to_csv(os.path.join(csvsavepath,savename+'.csv'))

