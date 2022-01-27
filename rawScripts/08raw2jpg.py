import numpy as np
import imageio

rawfile = np.fromfile(r'D:\Project\Python\Data\Data\sources\temp\6946\huijie\denoise\jbil\RGB_jbillsb5-101520.raw', dtype=np.uint8)  # 以float32读图片
print(rawfile.shape)
rawfile = rawfile.reshape(400, 400,3)
#rawfile.shape = (400, 400)
print(rawfile.shape)
b = rawfile.astype(np.uint8)  # 变量类型转换，float32转化为int8
print(b.dtype)
imageio.imwrite(r"D:\Project\Python\Data\Data\sources\temp\6946\huijie\denoise\jbil\RGB_jbillsb5-101520.jpg", b)

import matplotlib.pyplot as pyplot

pyplot.imshow(rawfile)
pyplot.show()
