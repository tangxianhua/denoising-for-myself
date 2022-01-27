import cv2
from matplotlib import pylab as plb
path1 = r'D:\Project\Python\Data\Data\sources\temp\200H0057.tif'
path2 = r'D:\Project\Python\Data\Data\sources\temp\200H0059.tif'
path3 = r'D:\Project\Python\Data\Data\sources\temp\200H0061.tif'
path4 = r'D:\Project\Python\Data\Data\sources\temp\200H0063.tif'
path5 = r'D:\Project\Python\Data\Data\sources\temp\200H0065.tif'
path6 = r'D:\Project\Python\Data\Data\sources\temp\200H0067.tif'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
img3 = cv2.imread(path3)
img4 = cv2.imread(path4)
img5 = cv2.imread(path5)
img6 = cv2.imread(path6)
add = cv2.addWeighted(img1, 0.5, img2, 0.5,0)
add2 = cv2.addWeighted(add,0.65,img3,0.35,0)
add3 = cv2.addWeighted(add2,0.75,img4,0.25,0)
add4 = cv2.addWeighted(add3,0.80,img5,0.20,0)
add5 = cv2.addWeighted(add4,0.84,img6,0.16,0)
cv2.imwrite('D:\Project\Python\Data\Data\sources\\temp\\all.jpg',add5)
plb.imshow(add5,'gray')
plb.show()