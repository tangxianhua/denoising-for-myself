import math
import numpy as np
import os
#
# #权重sigma设置
# # sigmalist = [[5,10,20],[5,10,25],[10,15,20],
# #              [10,15,25],[15,20,25],[35,40,1],
# #              [40,45,1],[45,50,1],[50,50,1]]
# sigmalist = [[20,25,30]]
# winsize = 3#窗口大小
# sigmalist = np.asarray(sigmalist)
# times = 100000#权重扩大的倍数1048576
# savepath = r'D:\Project\Done\Denoise220124\ref'
# gaussname = 'gaussAll'
# gaussNormname = 'gaussNormall'
# file1 = open(os.path.join(savepath,gaussname+'.txt'),'w')
# file2 = open(os.path.join(savepath,gaussNormname+'.txt'), 'w')
# for i in range(sigmalist.shape[0]):
#     # #窗口生成权重，便于查看
#     def getClosenessWeight(sigma_g,H,W):
#         r,c = np.mgrid[0:H:1,0:W:1]
#         r=r.astype(np.float64)
#         c=c.astype(np.float64)
#         r-=(H-1)/2
#         c-=(W-1)/2
#         closeWeight = np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigma_g,2))
#         return closeWeight*times
#     Weightr = getClosenessWeight(sigmalist[i][0],winsize,winsize)
#     Weightb = getClosenessWeight(sigmalist[i][1],winsize,winsize)
#     Weightg = getClosenessWeight(sigmalist[i][2],winsize,winsize)
#     #写入txt表
#     #写入R权重
#     for i in range(0,winsize):
#          for j in range(0,winsize):
#                  file1.write(str(int(Weightr[i][j])))
#                  file1.write('    ')
#     file1.write('\n')
#     #写入B权重
#     for i in range(0,winsize):
#          for j in range(0,winsize):
#                  file1.write(str(int(Weightb[i][j])))
#                  file1.write('    ')
#     file1.write('\n')
#     #写入G权重
#     for i in range(0,winsize):
#          for j in range(0,winsize):
#                  file1.write(str(int(Weightg[i][j])))
#                  file1.write('    ')
#     file1.write('\n')
#     #权重归一化
#     weightRNorm = (Weightr/Weightr.sum())*times
#     weightBNorm = (Weightb/Weightb.sum())*times
#     weightGNorm = (Weightg/Weightg.sum())*times
#     #归一化权重写入txt表
#     #写入R权重
#     for i in range(0,winsize):
#          for j in range(0,winsize):
#                  file2.write(str(int(weightRNorm[i][j])))
#                  file2.write('    ')
#     file2.write('\n')
#     #写入B权重
#     for i in range(0,winsize):
#          for j in range(0,winsize):
#                  file2.write(str(int(weightBNorm[i][j])))
#                  file2.write('    ')
#     file2.write('\n')
#     #写入G权重
#     for i in range(0,winsize):
#          for j in range(0,winsize):
#                  file2.write(str(int(weightGNorm[i][j])))
#                  file2.write('    ')
#     file2.write('\n')



#jbil权重查找表
listy = []
sigmar = 20
sigmab = 25
sigmag = 30
times = 1024
savepathname = os.path.join(r'D:\Project\Done\Denoise220124\ref','jbilLUT_'+str(sigmar)+'_'+str(sigmab)+'_'+str(sigmag)+'.txt')
#r的查找表
for i in range(0,4096):
        temp = ((i*i)/(sigmar*sigmar))*-0.5
        num = math.exp(temp)#exp(abs(f(i,j)-f(k,l))**2*-0.5)
        num = num * times
        if num >= 1:
        #     listy.append(int((num*1024)))
        #     listx.append(i)
          listy.append(int(num))
        else:
          listy.append(1)
#b的查找表
for i in range(0,4096):
        temp = ((i*i)/(sigmab*sigmab))*-0.5
        num = math.exp(temp)#exp(abs(f(i,j)-f(k,l))**2*-0.5)
        num = num * times
        if num >= 1:
          listy.append(int(num))
        else:
          listy.append(1)
#g的查找表
for i in range(0,4096):
        temp = ((i*i)/(sigmag*sigmag))*-0.5
        num = math.exp(temp)#exp(abs(f(i,j)-f(k,l))**2*-0.5)
        num = num * times
        if num >= 1:
          listy.append(int(num))
        else:
          listy.append(1)
file = open(savepathname,'w')
lenth = len(listy)
for i in range(lenth):
    file.write(str(int(listy[i])))
    file.write('\n')
file.close()
