import numpy as np
import os
from matplotlib import pylab as plb
path = r'D:\Project\Python\Data\Data\sources\temp'
imagename = 'WB_ce-'
imgr = np.load(os.path.join(path,imagename+'r.npy'))
imgr = imgr.astype(np.float64)
imgb = np.load(os.path.join(path,imagename+'b.npy'))
imgb = imgb.astype(np.float64)
imgg = np.load(os.path.join(path,imagename+'g1.npy'))
imgg = imgg.astype(np.float64)
plb.imshow(imgr,'gray')
plb.show()
newimage = np.zeros((2012,3024,3))
for i in range(2012):
    for j in range(3024):
      newimage[i][j][0] = int(imgr[i][j])
      newimage[i][j][1] = int(imgg[i][j])
      newimage[i][j][2] = int(imgb[i][j])
newimage = newimage.astype(np.uint16)
print(newimage.shape)
newimage.tofile(os.path.join(path,imagename+'3c16bit.raw'))



# imgr = np.load(r'D:\Project\Python\Data\Data\sources\temp\WB_ce-r.npy')
# print(imgr.shape)
# imgr = imgr.astype(np.float64)
# imgb = np.load(r'D:\Project\Python\Data\Data\sources\temp\WB_ce-b.npy')
# print(imgb.shape)
# imgb = imgb.astype(np.float64)
# imgg = np.load(r'D:\Project\Python\Data\Data\sources\temp\WB_ce-g1.npy')
# print(imgg.shape)
# imgg = imgg.astype(np.float64)
# newimage = np.zeros((3024,2012,3))
# print(newimage.shape)
# plb
# for i in range(2012):
#     for j in range(3024):
#       newimage[i][j][0] = int(imgr[i][j])
#       newimage[i][j][1] = int(imgg[i][j])
#       newimage[i][j][2] = int(imgb[i][j])
# newimage = newimage.astype(np.uint16)
# # print(newimage)
# print(newimage.shape)
# newimage.tofile(r'D:\Project\Python\Data\Data\sources\temp\WB_ce-3ch.raw')