import numpy as np



#raw转npy
height=6048#6048
width=4024#4024
#channel = 3
img=np.fromfile(r'D:\Project\Python\Data\Data\sources\temp\WB_ce.raw',dtype = 'uint16')
# print(img.shape)
print(img.max())
#imgData = img.reshape(width, height,channel)
imgData = img.reshape(width, height)
np.save(r'D:\Project\Python\Data\Data\sources\temp\WB_ce.npy',imgData)