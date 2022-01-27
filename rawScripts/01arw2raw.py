import glob
import rawpy
import numpy as np
import os
im = rawpy.imread(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin\DSC02028.ARW')
raw = im.raw_image#raw_image
print("111111")
# #保存
np.save(r'D:\Project\Python\Data\Data\sources\temp\DSC00240.npy',raw)
raw.tofile(r'D:\Project\Python\Data\Data\sources\temp\DSC00240.raw')

#批处理
# path = r'D:\Project\Python\Data\Data\sources\temp\ARW'
# npysavepath = r'D:\Project\Python\Data\Data\sources\temp\npy'
# rawsavepath = r'D:\Project\Python\Data\Data\sources\temp\raw'
# imglist = glob.glob(os.path.join(path,'*ARW'))
# imglist.sort()
# length = len(imglist)
# for i in range(length):
#     basename = imglist[i].split('\\')[-1]
#     savename = basename.split('.')[0]
#     image = rawpy.imread(imglist[i])
#     raw = image.raw_image
#     np.save(os.path.join(npysavepath,savename+'.npy'),raw)
#     raw.tofile(os.path.join(rawsavepath,savename+'.raw'))










