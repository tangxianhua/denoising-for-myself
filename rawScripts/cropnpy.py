import numpy as np
from matplotlib import pylab as plb
path = r"D:\Project\Python\Data\Data\sources\npy\1918.npy"
out = np.load(path)
#out2 = np.squeeze(out)
#print(out2.shape)
newnpy = out[2800:3200,2600:3000]
np.save(r"D:\Project\Python\Data\Data\sources\npy\1918-400.npy",newnpy)
plb.imshow(newnpy,'gray')
plb.show()