import matplotlib.pyplot as plt
import numpy as np

output_path = "C:/Users/576473/Desktop/DSB 2017/tutorial/"
imgs = np.load(output_path+'DSBImages.npy') #N x 1 x 512 x 512
#imgs = imgs[0]

for i in range(len(imgs)):
    print "image %d" % i
    fig,ax = plt.subplots(1,1,figsize=[8,8])
    ax.imshow(imgs[i,0],cmap='gray')
    plt.show()
    raw_input("hit enter to cont : ")