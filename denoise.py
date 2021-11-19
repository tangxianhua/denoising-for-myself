import numpy as np
import cv2
from matplotlib import pylab as plb
from PIL import Image
import scipy.misc
import time
import sys



path = r"D:\Project\Python\Data\Data\sources\npy\DSC00017.npy"#90是噪声图
out = np.load(path)
out = out.astype(np.float32)
# cv2.imwrite('D:\Project\Python\Data\Data\sources\denoise\gauss90.tif',out)

#高斯滤波
dst = cv2.GaussianBlur(out,(3,3),250)
dst = dst.astype(np.uint16)
np.save(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\DSC00017-guass-33250.npy',dst)
cv2.imwrite(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\DSC00017-guass-33250.tif',dst)
#双边滤波
# starttime = time.time()
# dst = cv2.bilateralFilter(src=out, d=3, sigmaColor=800, sigmaSpace=500)
# dst = dst.astype(np.uint16)
# endtime = time.time()
# print("time cost", starttime-endtime)
# cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bilateral\bil32020-cv2.tif',dst)
#np.save(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\BIL\1917-R-bil.npy',dst)

#联合双边滤波
# starttime = time.time()
# joint = cv2.GaussianBlur(out,(5,5),300)#联合双边滤波引导图 (src,kernelsize,sigmax(x方向方差))
# dst = cv2.ximgproc.jointBilateralFilter(joint,out,6,500,0) #(joint, src, d, sigmaColor, sigmaSpace)
# endtime = time.time()
# print("time cost", starttime-endtime)
# basic_PSNR = ComputePSNR(out, dst)
# print('The PSNR of basic image is {} dB.\n'.format(basic_PSNR))
# dst = dst.astype(np.uint16)
# #cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-jbil.tif',dst)
# np.save(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-jbil.npy',dst)

#导向滤波
# starttime = time.time()
# imgx=cv2.Sobel(out,-1,1,0,ksize=1)#sobel算子 (src, ddepth(output image depth), dx, dy,ksize=1, 3, 5, 7)
# imgy=cv2.Sobel(out,-1,0,1,ksize=1)
# fuse=cv2.addWeighted(imgx,0.5,imgy,0.5,0)##导向滤波sobel引导图
# #joint = cv2.GaussianBlur(out,(5,5),300)#导向滤波高斯引导图
# dst = cv2.ximgproc.guidedFilter(fuse,out,3,500,-1)  #-1代表和原图相同深度 (guide, src, radius(radius of Guided Filter), eps(colorspace),dDepth
# endtime = time.time()
# print("time cost", starttime-endtime)
# dst = dst.astype(np.uint16)
# #cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-guide.tif',dst)
# np.save(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-guide.npy',dst)

# plb.imshow(out, 'gray')
# plb.show()
