import cv2
import numpy as np
import math
import sys
import glob
from shutil import copyfile
import misc
from scipy.signal import correlate2d
from scipy.stats import signaltonoise
import config as c
import os
import sys
import pickle
import argparse


def mean_SSD(im1, im2, mask=None):
    im = np.subtract(im1,im2) 
    if not mask is None:
        im *= mask
    return (np.mean(np.multiply(im, im)))

def mean_SSD_subject(subject):
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)
    num_images = len(mask_names)

    M = np.zeros((num_images,num_images))
    for i in range(num_images):
        mask_name = mask_names[i]
        image_name = mask_name.replace("_mask", "")
        image1 = cv2.cvtColor( cv2.imread(image_name) ,cv2.COLOR_BGR2GRAY)
        mask1 = cv2.cvtColor( cv2.imread(mask_name) ,cv2.COLOR_BGR2GRAY)
        for j in range(i+1, num_images):
            mask_name = mask_names[j]
            image_name = mask_name.replace("_mask", "")
            image2 = cv2.cvtColor( cv2.imread(image_name) ,cv2.COLOR_BGR2GRAY)
            mask2 = cv2.cvtColor( cv2.imread(mask_name) ,cv2.COLOR_BGR2GRAY)
            image2 = np.subtract(image1,image2)            
            M[i,j] = M[j,i] = (np.mean(np.multiply(image2, image2)))

    return M



def transformPoint(point, warp_matrix):
    [x,y] = [warp_matrix[0][0]*point[0]+warp_matrix[0][1]*point[1]+warp_matrix[0][2], warp_matrix[1][0]*point[0]+warp_matrix[1][0]*point[1]+warp_matrix[1][2]]
    return [-1*x, -1*y]

def zero_divide(im1, im2):
    for x in range(im1.shape[0]):
        for y in range(im1.shape[1]):
            val = im2[x,y]
            if val == 0:
                im1[x,y] = 0
            else:
                im1[x,y] /= val
    return im1


def register(im1, im2, warp_mode = cv2.MOTION_TRANSLATION, number_of_iterations = 5000, termination_eps = 1e-10):
    cc = -1
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC ( im1, im2, warp_matrix, warp_mode, criteria)
    except:
        pass
    return (cc, warp_matrix)


def transform(im1, warp_matrix, dim, warp_mode = cv2.MOTION_TRANSLATION):
    warp_method = getattr(cv2, 'warpAffine')
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_method = getattr(cv2, 'warpPerspective')
    im1_aligned = warp_method(im1, warp_matrix, dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im1_aligned


def stitch(im1, im2, mask1=None, mask2=None, count1=None, count2=None, threshold=-1, warp_mode = cv2.MOTION_TRANSLATION):
    output = None
    output_mask = None
    output_count = None

    (cc, warp_matrix) = register(im1, im2, warp_mode)
    if cc < threshold:
        return [output, output_mask, output_count, cc]


    domasks = False
    if not mask1 is None and not mask2 is None:
        domasks = True
    if count1 is None:
        count1 = np.ones((im1.shape[0], im1.shape[1]), dtype=np.float32)
    if count2 is None:
        count2 = np.ones((im2.shape[0], im2.shape[1]), dtype=np.float32)

     


    id_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    warp_matrix_orig = np.copy(id_matrix)

    p = [0, 0]
    [x, y] = (transformPoint(p, warp_matrix))

    img1_warp = False
    if x<0 :
        warp_matrix_orig[0][2]=-warp_matrix[0][2]
        warp_matrix[0][2]=0
    if y<0 :
        warp_matrix_orig[1][2]=-warp_matrix[1][2]
        warp_matrix[1][2]=0
    [x, y] = (transformPoint(p, warp_matrix))





    warp1 = (np.sum(warp_matrix_orig==id_matrix)<6)
    warp2 = (np.sum(warp_matrix==id_matrix)<6)

    sz = im1.shape
    sz2 = im2.shape
    p = [0, 0]
    [x, y] = (transformPoint(p, warp_matrix_orig))
    dim1 = (math.ceil(sz[1]+x), math.ceil(sz[0]+y))
    [x, y] = (transformPoint(p, warp_matrix))
    dim2 = (math.ceil(sz2[1]+x), math.ceil(sz2[0]+y))
    dim=( max(dim1[0],dim2[0]), max(dim1[1],dim2[1]) )



    im1_aligned = im1
    im2_aligned = im2
    mask1_aligned = mask1
    mask2_aligned = mask2
    count1_aligned = count1
    count2_aligned = count2

    if warp1:
        im1_aligned = transform(im1, warp_matrix_orig, dim, warp_mode)
        count1_aligned = transform(count1, warp_matrix_orig, dim, warp_mode)
        if domasks:
            mask1_aligned = transform(mask1, warp_matrix_orig, dim, warp_mode)
            
    if warp2:
        im2_aligned = transform(im2, warp_matrix, dim, warp_mode)
        count2_aligned = transform(count2, warp_matrix, dim, warp_mode)
        if domasks:
            mask2_aligned = transform(mask2, warp_matrix, dim, warp_mode)
        
    output_count = np.zeros((dim[1], dim[0]), dtype=np.float32)
    output_count[0:count1_aligned.shape[0], 0:count1_aligned.shape[1]] += count1_aligned 
    output_count[0:count2_aligned.shape[0], 0:count2_aligned.shape[1]] += count2_aligned 



    output = np.zeros((dim[1], dim[0]), dtype=np.float32)
    output[0:im1_aligned.shape[0], 0:im1_aligned.shape[1]] += im1_aligned * count1_aligned
    output[0:im2_aligned.shape[0], 0:im2_aligned.shape[1]] += im2_aligned * count2_aligned
    output = np.round( zero_divide(output ,output_count) )


    if domasks:
        output_mask = np.zeros((dim[1], dim[0]), dtype=np.float32)
        output_mask[0:mask1_aligned.shape[0], 0:mask1_aligned.shape[1]] += mask1_aligned * count1_aligned
        output_mask[0:mask2_aligned.shape[0], 0:mask2_aligned.shape[1]] += mask2_aligned * count2_aligned
        output_mask = np.round( zero_divide(output_mask,output_count) )
        output_mask = output_mask.astype(np.uint8)

    output = output.astype(np.uint8)
    output_count = output_count.astype(np.uint8)

    return [output, output_mask, output_count, cc]



def stitch_subject(subject, output_name, output_mask_name, output_count_name):
    threshold = 0.8;
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)

    part = 1
    while len(mask_names) > 0:
        mask_name = mask_names.pop(0)
        image_name = mask_name.replace("_mask", "")
        output = cv2.cvtColor( cv2.imread(image_name) ,cv2.COLOR_BGR2GRAY)
        output_mask = cv2.cvtColor( cv2.imread(mask_name) ,cv2.COLOR_BGR2GRAY)
        output_count = output * 0 + 1

        while len(mask_names) > 0:
            flag = True
            for m in range(len(mask_names)):
                mask_name = mask_names[m]
                image_name = mask_name.replace("_mask", "")
                image = cv2.cvtColor( cv2.imread(image_name) ,cv2.COLOR_BGR2GRAY)
                mask = cv2.cvtColor( cv2.imread(mask_name) ,cv2.COLOR_BGR2GRAY)
                count = image * 0 + 1
                [output_new, output_mask_new, output_count_new, cc] = stitch(image, output, mask, output_mask, count, output_count, threshold)
                if cc >= threshold:
                    [output, output_mask, output_count] = [output_new, output_mask_new, output_count_new]
                    mask_names.pop(m)
                    flag = False
                    print("appending "+image_name+" with mean SSD "+str(output_SSD))

                    cv2.imwrite( output_name, output )
                    cv2.imwrite( output_mask_name, output_mask )
                    cv2.imwrite( output_count_name, output_count )
                    break
            if flag: break

        part_output_name = output_name.replace(".tif", "_"+str(part)+".tif")
        part_output_mask_name = output_mask_name.replace(".tif", "_"+str(part)+".tif")
        part_output_count_name = output_count_name.replace(".tif", "_"+str(part)+".tif")
        cv2.imwrite( part_output_name, output )
        cv2.imwrite( part_output_mask_name, output_mask )
        cv2.imwrite( part_output_count_name, output_count )
        part += 1



def cleanup_subject(subject, output_folder, threshold = 0.8):
    overlap_factor = 0.2
    
    warp_mode = cv2.MOTION_TRANSLATION
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)
    image_names = [w.replace('_mask', '') for w in mask_names]

    pos = {}
    for m in range(len(mask_names)):
        mask = cv2.cvtColor( cv2.imread(mask_names[m]), cv2.COLOR_BGR2GRAY)
        pos[m] = (mask.sum()>0)
    
    copyover = []
    for m in range(len(image_names)):
        image_name = image_names[m]
        if pos[m]: 
            copyover.append(image_name)
            continue
        # print(image_name)

        image1 = cv2.cvtColor( cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
        dim = (image1.shape[1], image1.shape[0])
        flag = False
        for m2 in range(len(image_names)):
            if m==m2 or not pos[m2]: 
                continue 

            image_name2 = image_names[m2]

            #extra
            base1 = os.path.basename(image_name).split('.')[0] 
            base2 = os.path.basename(image_name2).split('.')[0]
            fnp = "warps/pass/"+base1+"-"+base2+".pickle"
            if not os.path.exists(fnp): continue
            with open(fnp, 'rb') as re:
                [warp_matrix, cc, overlap] = pickle.load(re)
            if cc >= threshold and overlap < overlap_factor:
                flag = True
                break;
            #extra

            # image2 = cv2.cvtColor( cv2.imread(image_name2) ,cv2.COLOR_BGR2GRAY)
            # (cc, warp_matrix) = register(image1, image2, warp_mode)
            # if cc >= threshold and abs(warp_matrix[0][2]) < dim[0]*overlap_factor and abs(warp_matrix[1][2]) < dim[1]*overlap_factor:
            #     flag = True
            #     print("   "+image_name2+"   "+str(cc)+"   "+str(max(abs(warp_matrix[0][2])/dim[0], abs(warp_matrix[1][2])/dim[1])) )
            #     break;

        if not flag:
            copyover.append(image_name)
    
    for image_name in copyover:
        mask_name = image_name.replace(".tif","_mask.tif")
        new_image_name = output_folder+"/"+os.path.basename(image_name)
        new_mask_name = output_folder+"/"+os.path.basename(mask_name)
        copyfile(image_name, new_image_name)
        copyfile(mask_name, new_mask_name)



def cleanup(output_folder, threshold = 0.8):
    mask_names = glob.glob('train/*_mask.tif')
    subjects = misc.uniq([i.split("/")[1].split("_")[0] for i in mask_names])
    misc.sort_nicely(subjects)
    for subject in subjects:
        print("subject "+str(subject))
        cleanup_subject(subject, output_folder, threshold = threshold)





def cleanup_subject2(subject, output_folder, threshold = 0.8, dice_threshold = 0.6):
    from train import dice_predict
    overlap_factor = 0.2
    
    warp_mode = cv2.MOTION_TRANSLATION
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)
    image_names = [w.replace('_mask', '') for w in mask_names]

    pos = {}
    for m in range(len(mask_names)):
        mask = cv2.cvtColor( cv2.imread(mask_names[m]), cv2.COLOR_BGR2GRAY)
        pos[m] = (mask.sum()>0)
    
    copyover = []
    for m in range(len(image_names)):
        image_name = image_names[m]
        if not pos[m]: 
            copyover.append(image_name)
            continue
        # print(image_name)

        image1 = cv2.cvtColor( cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
        mask_name = image_name.replace(".tif","_mask.tif")
        mask1 = cv2.cvtColor( cv2.imread(mask_name) ,cv2.COLOR_BGR2GRAY)

        dim = (image1.shape[1], image1.shape[0])
        num_close = 0
        dice = 0
        for m2 in range(len(image_names)):
            if m==m2 or not pos[m2]: 
                continue 

            image_name2 = image_names[m2]

            #extra
            base1 = os.path.basename(image_name).split('.')[0] 
            base2 = os.path.basename(image_name2).split('.')[0]
            fnp = "warps/pass/"+base1+"-"+base2+".pickle"
            if not os.path.exists(fnp): continue
            with open(fnp, 'rb') as re:
                [warp_matrix, cc, overlap] = pickle.load(re)
            if cc >= threshold and overlap < overlap_factor:
            #extra

            # image2 = cv2.cvtColor( cv2.imread(image_name2) ,cv2.COLOR_BGR2GRAY)
            # (cc, warp_matrix) = register(image1, image2, warp_mode)
            # if cc >= threshold and abs(warp_matrix[0][2]) < dim[0]*overlap_factor and abs(warp_matrix[1][2]) < dim[1]*overlap_factor:

                mask_name2 = image_name2.replace(".tif","_mask.tif")
                mask2 = cv2.cvtColor( cv2.imread(mask_name2) ,cv2.COLOR_BGR2GRAY)
                mask2_warped = cv2.warpAffine(mask2, warp_matrix, dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                dice_mask = dice_predict(mask1, mask2_warped)
                dice += dice_mask
                num_close += 1

        if num_close > 0 :
            dice/=num_close
            # print(image_name+" "+str(dice))
            if abs(dice) > dice_threshold:
                copyover.append(image_name)
    
    for image_name in copyover:
        mask_name = image_name.replace(".tif","_mask.tif")
        new_image_name = output_folder+"/"+os.path.basename(image_name)
        new_mask_name = output_folder+"/"+os.path.basename(mask_name)
        copyfile(image_name, new_image_name)
        copyfile(mask_name, new_mask_name)



def cleanup2(output_folder, threshold = 0.8):
    mask_names = glob.glob('train/*_mask.tif')
    subjects = misc.uniq([i.split("/")[1].split("_")[0] for i in mask_names])
    misc.sort_nicely(subjects)
    for subject in subjects:
        print("subject "+str(subject))
        cleanup_subject2(subject, output_folder, threshold = threshold)






def fixup_subject(subject, output_folder, threshold = 0.8):
    overlap_factor = 0.2
    
    warp_mode = cv2.MOTION_TRANSLATION
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)
    image_names = [w.replace('_mask', '') for w in mask_names]

    pos = {}
    for m in range(len(mask_names)):
        mask = cv2.cvtColor( cv2.imread(mask_names[m]), cv2.COLOR_BGR2GRAY)
        pos[m] = (mask.sum()>0)
    
    copyover = []
    for m in range(len(image_names)):
        image_name = image_names[m]
        if pos[m]: 
            copyover.append(image_name)
            continue

        image1 = cv2.cvtColor( cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
        mask1 = image1*0
        bestcc = 0
        dim = (image1.shape[1], image1.shape[0])
        flag = False
        for m2 in range(len(image_names)):
            if m==m2 or not pos[m2]: 
                continue 

            image_name2 = image_names[m2]
            image2 = cv2.cvtColor( cv2.imread(image_name2) ,cv2.COLOR_BGR2GRAY)
            mask2 = cv2.cvtColor( cv2.imread(image_name2.replace(".tif","_mask.tif")) ,cv2.COLOR_BGR2GRAY)

            #extra
            base1 = os.path.basename(image_name).split('.')[0] 
            base2 = os.path.basename(image_name2).split('.')[0]
            fnp = "warps/pass/"+base1+"-"+base2+".pickle"
            if not os.path.exists(fnp): continue
            with open(fnp, 'rb') as re:
                [warp_matrix, cc, overlap] = pickle.load(re)
            if cc >= threshold and overlap < overlap_factor:
            #extra

            # (cc, warp_matrix) = register(image1, image2, warp_mode)
            # if cc >= threshold and abs(warp_matrix[0][2]) < dim[0]*overlap_factor and abs(warp_matrix[1][2]) < dim[1]*overlap_factor:
                flag = True
                if bestcc < cc:
                    bestcc = cc
                    mask1 = cv2.warpAffine(mask2, warp_matrix, dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        if not flag:
            copyover.append(image_name)
        else:
            mask_name = image_name.replace(".tif","_mask.tif")
            new_image_name = output_folder+"/"+os.path.basename(image_name)
            new_mask_name = output_folder+"/"+os.path.basename(mask_name)
            copyfile(image_name, new_image_name)
            cv2.imwrite( new_mask_name, mask1)
    

    for image_name in copyover:
        mask_name = image_name.replace(".tif","_mask.tif")
        new_image_name = output_folder+"/"+os.path.basename(image_name)
        new_mask_name = output_folder+"/"+os.path.basename(mask_name)
        copyfile(image_name, new_image_name)
        copyfile(mask_name, new_mask_name)



def fixup(output_folder, threshold = 0.8):
    mask_names = glob.glob('train/*_mask.tif')
    subjects = misc.uniq([i.split("/")[1].split("_")[0] for i in mask_names])
    misc.sort_nicely(subjects)
    for subject in subjects:
        print("subject "+str(subject))
        fixup_subject(subject, output_folder, threshold = threshold)




def find_non_overlapping_subject(subject, output_folder, threshold = 0.92):
    overlap_factor = 0.2
    warp_mode = cv2.MOTION_TRANSLATION
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)
    
    non_overlapping = []
    while len(mask_names) > 0:
        mask_name = mask_names.pop(0)
        image_name = mask_name.replace('_mask', '')
        image1 = cv2.cvtColor( cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
        mask1 = cv2.cvtColor( cv2.imread(mask_name), cv2.COLOR_BGR2GRAY)
        dim = (image1.shape[1], image1.shape[0])
        bestsnr = signaltonoise(image1, axis=None)
        bestimage = image_name
        pos_mask = (mask1.sum()>0)
        # if pos_mask:
        #     non_overlapping.append(image_name)
        #     continue
        
        for m in range(len(mask_names)-1,-1,-1):
            mask_name = mask_names[m]
            image_name2 = mask_name.replace('_mask', '')
            image2 = cv2.cvtColor( cv2.imread(image_name2) ,cv2.COLOR_BGR2GRAY)

            # (cc, warp_matrix) = register(image1, image2, warp_mode)
            # if cc >= threshold and abs(warp_matrix[0][2]) < dim[0]*overlap_factor and abs(warp_matrix[1][2]) < dim[1]*overlap_factor:

            #extra
            base1 = os.path.basename(image_name).split('.')[0] 
            base2 = os.path.basename(image_name2).split('.')[0]
            fnp = "warps/pass/"+base1+"-"+base2+".pickle"
            if not os.path.exists(fnp): continue
            with open(fnp, 'rb') as re:
                [warp_matrix, cc, overlap] = pickle.load(re)
            if cc >= threshold and overlap < overlap_factor:
            #extra

                mask2 = cv2.cvtColor( cv2.imread(mask_name) ,cv2.COLOR_BGR2GRAY)
                pos_mask2 = (mask2.sum()>0)
                mask_names.pop(m)
                snr = signaltonoise(image2, axis=None)
                if (pos_mask==0 and pos_mask2==1) or (pos_mask==pos_mask2 and snr > bestsnr):
                    bestsnr = snr
                    bestimage = image_name2
                    pos_mask = pos_mask2
        non_overlapping.append(bestimage)
    
    for image_name in non_overlapping:
        mask_name = image_name.replace(".tif","_mask.tif")
        new_image_name = output_folder+"/"+os.path.basename(image_name)
        new_mask_name = output_folder+"/"+os.path.basename(mask_name)
        copyfile(image_name, new_image_name)
        copyfile(mask_name, new_mask_name)



def find_non_overlapping(output_folder, threshold = 0.92):
    mask_names = glob.glob('train/*_mask.tif')
    subjects = misc.uniq([i.split("/")[1].split("_")[0] for i in mask_names])
    misc.sort_nicely(subjects)
    for subject in subjects:
        print("subject "+str(subject))
        find_non_overlapping_subject(subject, output_folder, threshold = threshold)



def find_non_overlapping_subject2(subject, output_folder, threshold = 0.92):
    overlap_factor = 0.2
    warp_mode = cv2.MOTION_TRANSLATION
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    mask_names = glob.glob('train/'+str(subject)+'_*_mask.tif')
    misc.sort_nicely(mask_names)
    
    non_overlapping = []
    while len(mask_names) > 0:
        mask_name = mask_names.pop(0)
        image_name1 = mask_name.replace('_mask', '')
        image1 = cv2.cvtColor( cv2.imread(image_name1), cv2.COLOR_BGR2GRAY)
        mask1 = cv2.cvtColor( cv2.imread(mask_name), cv2.COLOR_BGR2GRAY)
        dim = (image1.shape[1], image1.shape[0])
        snr1 = signaltonoise(image1, axis=None)

        # pos_mask = (mask1.sum()>0)
        # if pos_mask:
        #     non_overlapping.append(image_name1)
        #     continue

        flag = True
        for m in range(len(mask_names)-1,-1,-1):
            mask_name = mask_names[m]
            image_name2 = mask_name.replace('_mask', '')
            image2 = cv2.cvtColor( cv2.imread(image_name2) ,cv2.COLOR_BGR2GRAY)

            #extra
            base1 = os.path.basename(image_name1).split('.')[0] 
            base2 = os.path.basename(image_name2).split('.')[0]
            fnp = "warps/pass/"+base1+"-"+base2+".pickle"
            if not os.path.exists(fnp): continue
            with open(fnp, 'rb') as re:
                [warp_matrix, cc, overlap] = pickle.load(re)
            if cc >= threshold and overlap < overlap_factor:
            #extra


            # (cc, warp_matrix) = register(image1, image2, warp_mode)
            # if cc >= threshold and abs(warp_matrix[0][2]) < dim[0]*overlap_factor and abs(warp_matrix[1][2]) < dim[1]*overlap_factor:
                snr2 = signaltonoise(image2, axis=None)
                if snr1 > snr2:
                    mask_names.pop(m)
                    # print("   rm "+image_name2)
                else:
                    # print("   rm "+image_name1)
                    flag = False
                    break
            
        if flag:
            non_overlapping.append(image_name1)
    
    for image_name in non_overlapping:
        mask_name = image_name.replace(".tif","_mask.tif")
        new_image_name = output_folder+"/"+os.path.basename(image_name)
        new_mask_name = output_folder+"/"+os.path.basename(mask_name)
        copyfile(image_name, new_image_name)
        copyfile(mask_name, new_mask_name)



def find_non_overlapping2(output_folder, threshold = 0.92):
    mask_names = glob.glob('train/*_mask.tif')
    subjects = misc.uniq([i.split("/")[1].split("_")[0] for i in mask_names])
    misc.sort_nicely(subjects)
    for subject in subjects:
        print("subject "+str(subject))
        find_non_overlapping_subject2(subject, output_folder, threshold = threshold)



def register_all_to_all(img_folder='train', output_folder='warps', source=None, persubject=True):
    threshold = 0.8
    overlap_factor = 0.2
    warp_mode = cv2.MOTION_TRANSLATION
    #output_folder_all = output_folder+"/all"
    output_folder_pass = output_folder+"/pass"
    output_folder_show = output_folder+"/pass-show"
    #if not os.path.exists(output_folder_all):
    #    os.makedirs(output_folder_all)
    if not os.path.exists(output_folder_pass):
        os.makedirs(output_folder_pass)
    #if not os.path.exists(output_folder_show):
    #    os.makedirs(output_folder_show)
    
    all_names = glob.glob(img_folder+"/*tif")
    image_names = []
    for name in all_names:
        if 'mask' not in name:
            image_names.append(name)
    misc.sort_nicely(image_names)

    if source is None:
        source = image_names
    else:
        source = [source]
        
    for image_name in source:
        print(image_name)
        image1 = cv2.cvtColor( cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
        base1 = os.path.basename(image_name).split('.')[0]
        dim = (image1.shape[1], image1.shape[0])

        # fno = output_folder_show+"/"+base1+".tif"
        #cv2.imwrite( fno, image1 )
        
        for image_name2 in image_names:
            if image_name == image_name2: continue
            base2 = os.path.basename(image_name2).split('.')[0]
            if persubject:
                if base1.split('_')[0] != base2.split('_')[0]: continue
            #fn = output_folder_all+"/"+base1+"-"+base2+".pickle"
            #if os.path.exists(fn): continue

            print("   "+image_name2)
            image2 = cv2.cvtColor( cv2.imread(image_name2) ,cv2.COLOR_BGR2GRAY)

            (cc, warp_matrix) = register(image1, image2, warp_mode)
            overlap = max( abs(warp_matrix[0][2])/dim[0], abs(warp_matrix[1][2])/dim[1] )
            warp = [warp_matrix, cc, overlap]

            #with open(fn, 'wb') as wr:
            #    pickle.dump(warp, wr)

            if cc >= threshold and overlap < overlap_factor:
                fnp = output_folder_pass+"/"+base1+"-"+base2+".pickle"
                with open(fnp, 'wb') as wr:
                    pickle.dump(warp, wr)

                #fnt = output_folder_show+"/"+base1+"-"+base2+".tif"
                #image2_warped = cv2.warpAffine(image2, warp_matrix, dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # image2_warped = transform(image2, warp_matrix, image1.shape)
                #cv2.imwrite( fnt, image2_warped )




def evaluate_overlap(img_folder='train', warp_folder='warps', source=None):
    import train
    warp_mode = cv2.MOTION_TRANSLATION
    warp_folder_pass = warp_folder+"/pass"

    if source is None:
        source = glob.glob(img_folder+"/*mask.tif")
        misc.sort_nicely(source)
    else:
        source = [source]
    
    for mask_name in source:
        dice = 0
        mask1 = cv2.cvtColor( cv2.imread(mask_name), cv2.COLOR_BGR2GRAY)
        base1 = os.path.basename(mask_name).split('.')[0].replace("_mask","")
        dim = (mask1.shape[1], mask1.shape[0])

        close_warp_names = glob.glob(warp_folder_pass+"/"+base1+"-*.pickle")
        # print(warp_folder_pass+"/"+base1+"-*.pickle")
        if len(close_warp_names)==0:
            continue

        print(mask_name)
        num_close = 0
        for warp in close_warp_names:
            base2 = os.path.basename(warp).split('.')[0].split('-')[1]
            mask_name2 = img_folder+"/"+base2+"_mask.tif"
            if not os.path.exists(mask_name2):
                continue

            mask2 = cv2.cvtColor( cv2.imread(mask_name2) ,cv2.COLOR_BGR2GRAY)
            with open(warp, 'rb') as re:
                [warp_matrix, cc, overlap] = pickle.load(re)

            mask2_warped = cv2.warpAffine(mask2, warp_matrix, dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            dice_mask = train.dice_predict(mask1, mask2_warped)
            print("   "+mask_name2+": "+str(dice_mask))
            dice += dice_mask
            num_close += 1
        dice /= num_close
        print("overall: "+str(dice))







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-clean", dest="clean",  help="clean", default=0.0)
    parser.add_argument("-clean2", dest="clean2",  help="clean2", default=0.0)
    parser.add_argument("-fix", dest="fix",  help="fix", default=0.0)
    parser.add_argument("-nonover", dest="nonover",  help="nonover", default=0.0)
    parser.add_argument("-nonover2", dest="nonover2",  help="nonover2", default=0.0)
    options = parser.parse_args()
    
    clean = float(options.clean)
    clean2 = float(options.clean2)
    fix = float(options.fix)
    nonover = float(options.nonover)
    nonover2 = float(options.nonover2)
    if clean > 0:
        cleanup("train-clean-"+str(clean), threshold=clean)
    elif clean2 > 0:
        cleanup2("train-clean2-"+str(clean2), threshold=clean2)
    elif fix > 0:
        fixup("train-fix-"+str(fix), threshold=fix)
    elif nonover > 0:
        find_non_overlapping("train-nonover-"+str(nonover), threshold=nonover)
    elif nonover2 > 0:
        find_non_overlapping2("train-nonover2-"+str(nonover2), threshold=nonover2)

    # evaluate_overlap()
    # cleanup("train-clean-0.3", overlap_factor=0.3)

    # register_all_to_all(source=source)
    # evaluate_overlap(source=source)


# find_non_overlapping("train-reduced")

# input1_name = "train/1_1.tif"
# input2_name = "train/1_3.tif"
# mask1_name = "train/1_1_mask.tif"
# mask2_name = "train/1_3_mask.tif"
# image1 = cv2.cvtColor( cv2.imread(input1_name) ,cv2.COLOR_BGR2GRAY)
# mask1  = cv2.cvtColor( cv2.imread(mask1_name) ,cv2.COLOR_BGR2GRAY)
# image2 = cv2.cvtColor( cv2.imread(input2_name) ,cv2.COLOR_BGR2GRAY)
# mask2  = cv2.cvtColor( cv2.imread(mask2_name) ,cv2.COLOR_BGR2GRAY)
# [output, output_mask, output_count] = stitch(image1, image2, mask1, mask2)
# cv2.imwrite( output_name, output )
# cv2.imwrite( output_mask_name, output_mask )
# cv2.imwrite( output_count_name, output_count )


# output_name = "test.tif"
# output_mask_name = "test-mask.tif"
# output_count_name = "test-counts.tif"
# stitch_subject(1, output_name, output_mask_name, output_count_name)