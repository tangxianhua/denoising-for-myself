#联合双边滤波
#图像高斯平滑，近似性权重由高斯平滑后确定
import numpy as np
from scipy import signal
import cv2,math
import os
import time
from matplotlib import pylab as plb
def getClosenessWeight(sigma_g,H,W):
    r,c = np.mgrid[0:H:1,0:W:1]
    r=r.astype(np.float64)
    c=c.astype(np.float64)
    r-=(H-1)/2
    c-=(W-1)/2
    closeWeight = np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigma_g,2))
    return closeWeight
def jointBLF(I,H,W,sigma_g,sigma_d,borderType=cv2.BORDER_DEFAULT):
    closenessWeight = getClosenessWeight(sigma_g,H,W)#高斯距离权重
    #高斯平滑
    I = I.astype(np.uint16)
    #Ig = cv2.medianBlur(I,3)#中值滤波,数据类型必须为整型
    Ig = cv2.GaussianBlur(I,(W,H),sigma_g)#引导图
    cH = int((H-1)/2)
    cW = int((W-1)/2)
    Ip = cv2.copyMakeBorder(I,cH,cH,cW,cW,borderType)#噪声图填充高斯核的一半
    Igp = cv2.copyMakeBorder(Ig,cH,cH,cW,cW,borderType)#高斯滤波图填充高斯核一半
    rows,cols = I.shape
    i,j = 0,0
    jblf = np.zeros(I.shape,np.float64)
    for r in np.arange(cH,cH+rows,1):
        for c in np.arange(cW,cW+cols,1):
            pixel = Igp[r][c]
            rTop,rBottom = r-cH,r+cH
            cLeft,cRight = c-cW,c+cW
            region = Igp[rTop:rBottom+1,cLeft:cRight+1]
            similarityWeight = np.exp(-0.5*np.power(region-pixel,2.0)/math.pow(sigma_d,2.0))
            weight = closenessWeight*similarityWeight
            weight = weight/np.sum(weight)
            jblf[i][j] = np.sum(Ip[rTop:rBottom+1,cLeft:cRight+1]*weight)
            j+=1
        j = 0
        i+=1
    return jblf
if __name__ =='__main__':
    # starttime = time.time()
    # I = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\origin\DSC00031-r.npy')
    # fI = I.astype(np.float64)
    # jblf = jointBLF(fI,5,5,15,15)#(I,H,W,sigma_g,sigma_d,borderType=cv2.BORDER_DEFAULT) H和W也是高斯模糊的核大小，也是权重计算核的大小
    # endtime = time.time()
    # print("cost",endtime-starttime)
    # jblfint16 = jblf.astype(np.uint16)
    # np.save(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\30\jbil\DSC00031-02-g2.npy',jblfint16)
    # jblfint16.tofile(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\30\jbil\DSC00031-02-g2.raw')


    path = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\origin'
    savepath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\guidejbil\jbil'
    imagename = 'DSC00039-'
    num = '07-'
    winsize = 3
    print('1')
    sigmad = 15
    sigmar = 15
    I = np.load(os.path.join(path,imagename+'r.npy'))
    fI = I.astype(np.float64)
    jblf = jointBLF(fI,winsize,winsize,sigmad,sigmar)
    jblfint16 = jblf.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+num+'r.npy'),jblfint16)
    jblfint16.tofile(os.path.join(savepath,imagename+num+'r.raw'))

    print('2')
    sigmad = 20
    sigmar = 20
    I = np.load(os.path.join(path,imagename+'b.npy'))
    fI = I.astype(np.float64)
    jblf = jointBLF(fI,winsize,winsize,sigmad,sigmar)
    jblfint16 = jblf.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+num+'b.npy'),jblfint16)
    jblfint16.tofile(os.path.join(savepath,imagename+num+'b.raw'))
    #
    print('3')
    sigmad = 30
    sigmar = 30
    I = np.load(os.path.join(path,imagename+'g1.npy'))
    fI = I.astype(np.float64)
    jblf = jointBLF(fI,winsize,winsize,sigmad,sigmar)
    jblfint16 = jblf.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+num+'g1.npy'),jblfint16)
    jblfint16.tofile(os.path.join(savepath,imagename+num+'g1.raw'))

    print('4')
    sigmad = 30
    sigmar = 30
    I = np.load(os.path.join(path,imagename+'g2.npy'))
    fI = I.astype(np.float64)
    jblf = jointBLF(fI,winsize,winsize,sigmad,sigmar)
    jblfint16 = jblf.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+num+'g2.npy'),jblfint16)
    jblfint16.tofile(os.path.join(savepath,imagename+num+'g2.raw'))