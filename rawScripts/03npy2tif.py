import numpy as np
import cv2

#npy保存tif
path1 = r'D:\Project\Python\Data\Data\sources\npy\DSC00017.npy'
img = np.load(path1)
cv2.imwrite(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\DSC00017.tif',img)