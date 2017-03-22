# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:36:57 2017

@author: 572203
"""

'''
What's up fools, I annotated a few lines code at the beginning so that we can work on only one patient;
once we get one patient's lungs segmented, we can delete the lines and generalize the for loop to run on
all our DSB patients. To summarize, I combined the two giant for loops together from the original LUNA_segment_lung_ROI.py
file.

Since we talked earlier today, I managed to fix the problem; the segmented_lungs array at the end of the code has all the patient's
CT slices with the lung segmentation code applied to them.  However, as well know, not all the CT slices have lungs in them; and yet,
the number of slices hasn't decreased.  Therefore, we need to come up with a way to remove the slices that don't have real lungs in them!!
I think this can be accomplished, or maybe the method is already in the code and I've messed up.  In either case, let's get cracking
on this so we can just have lungs!!
'''
import os
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from tqdm import tqdm
import dicom

working_path = "C:/Users/572203/Documents/Data Science Bowl 2017/stage1/stage1/" #this is the location of the DSB patient folders on your computer
save_path = "C:/Users/572203/Documents/Data Science Bowl 2017/stage1/stage1_tutorial/" #this is the location where the all-inclusive segmented lung DSB stage1 numpy array will go 
all_patients = sorted(os.listdir(working_path))

#for fcount, patient_folder in enumerate(tqdm(all_patients)):
    #dicom_files = os.listdir(working_path+patient_folder)
    dicom_files = os.listdir(working_path+all_patients[0]) #delete this line once we get the code working
    #dicom_images = [dicom.read_file(working_path+patient_folder+'/'+i) for i in dicom_files]
    dicom_images = [dicom.read_file(working_path+all_patients[0]+'/'+i) for i in dicom_files]#delete this line once we get the code working
    dicom_images.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(dicom_images[0].ImagePositionPatient[2] - dicom_images[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(dicom_images[0].SliceLocation - dicom_images[1].SliceLocation)
        
    for s in dicom_images:
        s.SliceThickness = slice_thickness
    
    imgs_to_process = np.stack([s.pixel_array for s in dicom_images])
    segmented_lungs = []
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        #Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400,100:400] 
        mean = np.mean(middle)  
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        img[img==max]=mean
        img[img==min]=mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid 
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region 
        # engulf the vessels and incursions into the lung cavity by 
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions 
        #  with a bbox that is to large in either dimnsion
        #  Also, the lungs are generally far away from the top 
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity. 
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        
        
        #This is the point in which I combined the two giant for loops in the LUNA_segment_lung_ROI.py file
        #so that we ignored all the redundant creation of .npy files.
        
        
        
        #at some point in here, if we want to, we could save the old images and the blank mask if we wanted, but there isn't really a reason.
        img= mask*img          # apply lung mask
        #
        # renormalizing the masked image (in the mask region)
        #
        new_mean = np.mean(img[mask>0])  
        new_std = np.std(img[mask>0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        old_min = np.min(img)       # background color
        img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
        img = img-new_mean
        img = img/new_std
        #make image bounding box  (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col-min_col
        height = max_row - min_row
        if width > height:
            max_row=min_row+width
        else:
            max_col = min_col+height
        # 
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        # 
        img = img[min_row:max_row,min_col:max_col]
        mask =  mask[min_row:max_row,min_col:max_col] # why is this here?
        if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no go0d regions
            pass
        else:
            # moving range to -1 to 1 to accomodate the resize function
            mean = np.mean(img)
            img = img - mean
            min = np.min(img)
            max = np.max(img)
            img = img/(max-min)
            new_img = resize(img,[512,512])
        segmented_lungs.append(new_img) 
    segmented_lungs = np.stack(segmented_lungs)
    #np.save(os.path.join(save_path+"DSBimages_"+str(fcount-1)), segmented_lungs)
    #need to add another block of code here that creates one file of all the slices, like LUNA script does
    
    # in the above code, could use np.savez here so you can save the numpy array as a compressed file.  Just a thought.
    #can also use hdf5 to store these data sets, which apparently is very efficient and can be sliced into without heavy
    #loading into RAM while the hdf5 file is just sitting there on your disk...

