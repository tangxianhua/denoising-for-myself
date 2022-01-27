import numpy as np
import cv2
import time
import os
def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑
    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差
    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    q = mean_a * I + mean_b



    print('q')
    print(q[2,2])

    return q


if __name__ == '__main__':
    # eps = 0.01
    # winSize = (5,5)
    # image = cv2.imread(r'./5921.png', cv2.IMREAD_ANYCOLOR)
    # image = cv2.resize(image, None,fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
    # I = image/255.0        #将图像归一化
    # p =I
    # guideFilter_img = guideFilter(I, p, winSize, eps)
    #
    # # 保存导向滤波结果
    # guideFilter_img  = guideFilter_img  * 255
    # guideFilter_img [guideFilter_img  > 255] = 255
    # guideFilter_img  = np.round(guideFilter_img )
    # guideFilter_img  = guideFilter_img.astype(np.uint8)
    # cv2.imshow("image",image)
    # cv2.imshow("winSize_5", guideFilter_img )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # eps = 0.001
    # winSize = (5, 5)
    # #winSize2 = (3, 3)
    # image = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\origin\DSC00031-g2.npy')
    # I = image/65535
    # starttime = time.time()
    # # 均值滤波引导
    # #p = cv2.blur(I, winSize2)
    # #img_median = cv2.medianBlur(image, 3)#img参数，应该是uint类型，中值滤波引导
    # #img_median = img_median/18000
    # #sobel引导图
    # # imgx=cv2.Sobel(I,-1,1,0,ksize=3)#sobel算子 (src, ddepth(output image depth), dx, dy,ksize=1, 3, 5, 7)
    # # imgy=cv2.Sobel(I,-1,0,1,ksize=3)
    # # fuse=cv2.addWeighted(imgx,1,imgy,1,0)
    # # cv2.imwrite(r'D:\Project\Python\Data\Data\sources\guide\sobel.tif',fuse)
    # #高斯滤波引导
    # #GaussianBlur = cv2.GaussianBlur(I, (3, 3), 200)# src, ksize, sigmaX标准差
    # #双边滤波引导
    # bilateral = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\30\jbil\DSC00031-01-g2.npy')
    # bilateral = bilateral/65535
    #
    # guideFilter_img = guideFilter(I,bilateral, winSize, eps)
    # # 细节增强
    # # result_enhanced = (I - guideFilter_img) * 10000 + guideFilter_img
    # # result_enhanced = cv2.normalize(result_enhanced, result_enhanced, 1, 0, cv2.NORM_MINMAX)
    #
    # guideFilter_img  = guideFilter_img  * 65535
    # guideFilter_img [guideFilter_img  > 65535] = 65535
    # guideFilter_img  = np.round(guideFilter_img )
    # endtime = time.time()
    # print('cost time',endtime-starttime)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\30\guide-jbil\DSC00031-12-g2.npy',guideFilter_img)
    # guideFilter_img.tofile(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\30\guide-jbil\DSC00031-12-g2.raw')

    epsr = 0.000007
    epsg = 0.00001
    epsb = 0.00001
    winSize = (3, 3)
    deno = 16383
    path = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin'
    guideimagepath = r'D:\Project\Python\Data\Data\sources\temp\2029\jbil'
    savepath = r'D:\Project\Python\Data\Data\sources\temp\2029\guide'
    imagename = 'DSC02029-'
    guidename = 'c-float-'
    savename = 'g01-'


    image = np.load(os.path.join(path,imagename+'r.npy'))
    print('RRRRRRRRRRRRRRRRRRRRRRRRR')
    #print(image[0:6,0:6])
    I = image / deno
    starttime = time.time()
    bilateral = np.load(os.path.join(guideimagepath,guidename+'r.npy'))
    bilateral = bilateral / deno
    guideFilter_img = guideFilter(I, bilateral, winSize, epsr)
    guideFilter_img = guideFilter_img * deno
    #print(guideFilter_img[3,3])
    guideFilter_img[guideFilter_img > deno] = deno
    guideFilter_img = np.round(guideFilter_img)
    #print(guideFilter_img[3, 3])
    # print(guideFilter_img[0:6, 0:6])
    # print('输出')
    endtime = time.time()
    print('cost time', endtime - starttime)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'r.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'r.raw'))

    image = np.load(os.path.join(path,imagename+'b.npy'))
    print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
    #print(image[0:6, 0:6])
    I = image / deno
    starttime = time.time()
    bilateral = np.load(os.path.join(guideimagepath,guidename+'b.npy'))
    bilateral = bilateral / deno
    guideFilter_img = guideFilter(I, bilateral, winSize, epsb)
    guideFilter_img = guideFilter_img * deno
    #print(guideFilter_img[2,2])
    guideFilter_img[guideFilter_img > deno] = deno
    guideFilter_img = np.round(guideFilter_img)
    #print(guideFilter_img[2, 2])
    endtime = time.time()
    print('cost time', endtime - starttime)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'b.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'b.raw'))

    image = np.load(os.path.join(path,imagename+'g1.npy'))
    print('G1G11G11111111111111111111111111111111')
    #print(image[0:6, 0:6])
    I = image / deno
    starttime = time.time()
    bilateral = np.load(os.path.join(guideimagepath,guidename+'g1.npy'))
    bilateral = bilateral / deno
    guideFilter_img = guideFilter(I, bilateral, winSize, epsg)
    guideFilter_img = guideFilter_img * deno

    guideFilter_img[guideFilter_img > deno] = deno
    guideFilter_img = np.round(guideFilter_img)
    #print(guideFilter_img[0:6, 0:6])
    endtime = time.time()
    print('cost time', endtime - starttime)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'g1.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'g1.raw'))

    image = np.load(os.path.join(path,imagename+'g2.npy'))
    print('G222222222222222222222222222222222222')
    #print(image[0:6, 0:6])
    I = image / deno
    starttime = time.time()
    bilateral = np.load(os.path.join(guideimagepath,guidename+'g2.npy'))
    bilateral = bilateral / deno
    guideFilter_img = guideFilter(I, bilateral, winSize, epsg)
    guideFilter_img = guideFilter_img * deno
    guideFilter_img[guideFilter_img > deno] = deno
    guideFilter_img = np.round(guideFilter_img)
    #print(guideFilter_img[0:6, 0:6])
    endtime = time.time()
    print('cost time', endtime - starttime)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'g2.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'g2.raw'))
