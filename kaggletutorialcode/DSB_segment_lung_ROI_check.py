import matplotlib.pyplot as plt
import numpy as np

output_path = "C:/Users/576473/Desktop/DSB 2017/tutorial/"
imgs = np.load(output_path+'DSBImages.npy')
imgs = imgs[0]

for i in range(len(imgs)):
    print "image %d" % i
    fig,ax = plt.subplots(1,1,figsize=[8,8])
    ax.imshow(imgs[i],cmap='gray')
    plt.show()
    raw_input("hit enter to cont : ")