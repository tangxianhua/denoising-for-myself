import numpy as np
from matplotlib import pylab as plb
import cv2
import os
heightall = 4024#4024
widthall = 6048#6048
height=3024
width=2012
path = r'D:\Project\Python\Data\Data\sources\temp'
imagename = 'WB_ce'
newname = 'WB_ce'
rawnpy  = np.load(os.path.join(path,imagename+'.npy'))
print(os.path.join(path,imagename+'.npy'))
#rawnpy  = np.load(r'D:\Project\Python\Data\Data\sources\temp\2029\guide\ceguidecefloat.npy')
#path = r'D:\Project\Python\Data\Data\sources\temp\2029\guide'
#imagename = 'c-float'

print(rawnpy.shape)
rnpy = np.zeros(6084288)#长的一半*宽的一半
print(rnpy.shape)
ii = 0
for i in range(0,heightall,2):
    for j in range(0,widthall,2):
        rnpy[ii] = rawnpy[i][j]*16
        ii += 1
rnpy = rnpy.astype(np.uint16)
imgData = rnpy.reshape(width, height)
np.save(os.path.join(path,newname+'-r.npy'),imgData)
imgData.tofile(os.path.join(path,newname+'-r.raw'))

ii = 0
for i in range(1,heightall,2):
    for j in range(1,widthall,2):
        rnpy[ii] = rawnpy[i][j]*16
        ii += 1
rnpy = rnpy.astype(np.uint16)
imgData = rnpy.reshape(width, height)
np.save(os.path.join(path,newname+'-b.npy'),imgData)
imgData.tofile(os.path.join(path,newname+'-b.raw'))

ii = 0
for i in range(0,heightall,2):
    for j in range(1,widthall,2):
        rnpy[ii] = rawnpy[i][j]*16
        ii += 1
rnpy = rnpy.astype(np.uint16)
imgData = rnpy.reshape(width, height)
np.save(os.path.join(path,newname+'-g1.npy'),imgData)
imgData.tofile(os.path.join(path,newname+'-g1.raw'))


ii = 0
for i in range(1,heightall,2):
    for j in range(0,widthall,2):
        rnpy[ii] = rawnpy[i][j]*16
        ii += 1
rnpy = rnpy.astype(np.uint16)
imgData = rnpy.reshape(width, height)
np.save(os.path.join(path,newname+'-g2.npy'),imgData)
imgData.tofile(os.path.join(path,newname+'-g2.raw'))