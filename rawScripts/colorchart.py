import numpy as np
import matplotlib.pylab as plt
import skimage.io as io
import cv2
# list1 = [[107,82,70],[184,146,129],[101,122,153],[95,107,69],[128,127,173],[129,188,171],\
#               [201,123,56],[77,92,166],[174,83,97],[86,61,104],[167,188,175],[213,160,55],\
#              [49,65,143],[99,148,80],[155,52,59],[227,197,52],[169,85,147],[61,135,167],\
#             [245,245,242],[200,201,201],[160,161,162],[120,120,121],[84,85,86],[52,53,54] ]

listApple = [[94, 63, 51], [183, 128, 109], [74, 103, 139], [73, 89, 48], [110, 108, 162], [84, 178, 155],\
             [211, 102, 30], [52, 71, 156], [180, 59, 79], [73, 42, 88], [145, 177, 39], [220, 143, 19],\
             [26, 47, 131], [60, 133, 54], [159, 29, 43], [232, 187, 0], [174, 60, 134], [0, 118, 154],\
             [242, 243, 239], [189, 191, 191], [144, 146, 146], [101, 102, 102], [65, 66, 68], [37, 37, 38]]
# listRGB1 = [[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],\
#              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63], [230,162,39],\
#              [35,63,147], [67,149,74], [180,49,57], [238,198,20], [193,84,151], [0,136,170],\
#              [245,245,243], [200,202,202], [161,163,163], [121,121,122], [82,84,86], [49,49,51]]
listgrey = [[0, 0, 0], [40, 40, 40], [80, 80, 80], [120, 120, 120], [160, 160, 160], [220, 220, 220],\
             [10, 10, 10], [50, 50, 50], [90, 90, 90], [130, 130, 130], [170, 170, 170], [220, 220, 220],\
             [20, 20, 20], [60, 60, 60], [100, 100, 100], [140, 140, 140], [180, 180, 180], [240, 240, 240],\
             [30, 30, 30], [70, 70, 70], [110, 110, 110], [150, 150, 150], [190, 190, 190], [255, 255, 255]]

b = np.array(listgrey)
image = np.zeros((4*150+5*20,6*150+7*20,3))
for i in range(4):
    for j in range(6):
        image[20+i*170: 20 +i*170+150,20+j*170:20+j*170+150,:] = b[i*6+j,:]
image =image.astype(int)
print(image.shape)
plt.imshow(image)
plt.show()
#plt.axis("off")
#image = image.astype(np.uint8)
#image.tofile(r'D:\Project\Python\Data\Data\sources\temp\chart.raw')
#cv2.imwrite(r'D:\Project\Python\Data\Data\sources\temp\chart.bmp',image)

