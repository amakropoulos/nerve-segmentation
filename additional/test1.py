import os
import sys
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import time
import math
import lasagne 
import scipy.misc
import argparse
import glob
from skimage import measure
import cv2
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter

import config as c
import model
import misc




# version=0
# depth=6
# filters=8
# testdir='test'
# submitdir='submit'

def postprocess(pred, minsize=5000, dilations=2, sigma=5):
    predthr = (pred[0]>=0.5)

    lcc = measure.label(predthr)
    num_lccs = np.max(lcc)
    if num_lccs>1:
        bestlcc=-1
        bestlccscore=-1
        for l in range(num_lccs):
            score = ((lcc == l+1)*pred).sum()
            if score > bestlccscore:
                bestlccscore=score
                bestlcc = l+1
        lcc=lcc.reshape(pred.shape)
        predthr=(lcc==bestlcc)

    if predthr.sum() <= minsize:
        predthr = np.zeros(pred.shape)
    else:
        predext = gaussian_filter(binary_dilation(predthr, iterations=dilations).astype(predthr.dtype), sigma)
        predthr =( ((predext>0.5)+predthr)>0 ).astype(predthr.dtype)

    return predthr


def join_models_cv(version=0, submitdir='submit', minsize=5000):
    start_time = time.clock()   
    print_dir = os.path.join(submitdir, 'v{}'.format(version))
    cv_dirs = glob.glob(print_dir+'/cv*')
    misc.sort_nicely(cv_dirs)
    num_images = len(glob.glob(cv_dirs[0]+'/*.tif'))
    num_cvs = len(cv_dirs)

    submission_file = os.path.join(submitdir, 'v{}.csv'.format(version))
    submission = open(submission_file,'w')
    submission.write('img,pixels\n')

    premod = 0
    for i in range(1,num_images+1):
        pred = None
        for d in range(num_cvs):
            img_name = cv_dirs[d]+'/'+str(i)+'_pred.tif'
            img = misc.load_image(img_name)

            # img *= weights[d] # weighted
            if d == 0:
                pred = img
            else:
                pred += img
        # pred /= weights_sum # weighted
        pred /= num_cvs

        pred = postprocess(pred, minsize=minsize)
        line = str(i)+","+misc.run_length(pred)
        submission.write(line+'\n')

        fnpred = os.path.join(print_dir, str(i)+'_pred.tif')
        scipy.misc.imsave(fnpred, pred.reshape(c.height, c.width) * np.float32(255))

        mod = math.floor(i / num_images * 10) 
        if mod > premod and mod > 0:
            print(str(mod*10)+'%..',end="",flush=True)
        premod = mod

    submission.close()
    print("Prediction took {:.3f}s".format(time.clock() - start_time))




def test_model(version=0, depth=6, filters=8, testdir='test', submitdir='submit', resize=0.0, probs_only=False, cv=0, minsize=5000, tta_num=1, seed=1234):
    modelversion, dtbased, datadir, drop, augment, shift, nonlinearity = model.version_parameters(version)

    start_time = time.clock()   
    shape = (1, 1, c.height, c.width)
    if resize:        
        orig_shape = shape
        shape = (1, 1, round(c.height*resize), round(c.width*resize) ) 
    # netshape = (tta_num, shape[1], shape[2], shape[3])
    netshape = shape

    input_var = T.tensor4('input')
    label_var = T.tensor4('label')
    net = model.network(input_var, netshape, version=modelversion, drop=drop, nonlinearity=nonlinearity, depth=depth, num_filters=filters)

    output_det = lasagne.layers.get_output(net['output'], deterministic=True)
    predict_model = theano.function(inputs=[input_var], outputs=output_det)

    misc.load_last_params(net['output'], version, best=True, cv=cv, seed=seed)

    print_idx = 0
    print_dir = os.path.join(submitdir, 'v{}'.format(version))
    if cv>0:
        print_dir = print_dir + "/cv{}".format(cv)
    if not os.path.exists(print_dir):
        os.makedirs(print_dir)
    if not probs_only:
        submission_file = os.path.join(submitdir, 'v{}.csv'.format(version))
        submission = open(submission_file,'w')
        submission.write('img,pixels\n')

    num_images = len(glob.glob(testdir+'/*.tif'))
    premod = 0

    for i in range(1,num_images+1):
        img_name = testdir+'/'+str(i)+'.tif'
        img = misc.load_image(img_name)

        if resize:
            img = cv2.resize(img[0], (shape[3], shape[2]), interpolation=cv2.INTER_CUBIC).reshape(shape)
        else:
            img = img.reshape(shape)


        if tta_num>1:
            start_time = time.clock()
            for tt in range(tta_num):
                if tt == 0:
                    tta_pred = predict_model(img)
                else:
                    tta_pred = misc.tta(img,predict_model)
                if resize:
                    tta_pred = cv2.resize(tta_pred[0][0], (orig_shape[3], orig_shape[2]), interpolation=cv2.INTER_LINEAR).reshape(orig_shape)
                if tt == 0:
                    pred = tta_pred
                else:
                    pred += tta_pred
            pred/=tta_num
            print("1 took " + str(time.clock()-start_time )+"s")

            # start_time = time.clock()
            # pred = misc.tta2(img, predict_model, tta_num, orig_shape)
            # print("2 took " + str(time.clock()-start_time )+"s")
        else:
            pred = predict_model(img)
            if resize:
                pred = cv2.resize(pred[0][0], (orig_shape[3], orig_shape[2]), interpolation=cv2.INTER_LINEAR).reshape(orig_shape)


        if not probs_only:
            pred = postprocess(pred, minsize=minsize)
            line = str(i)+","+misc.run_length(pred)
            submission.write(line+'\n')

        fnpred = os.path.join(print_dir, str(i)+'_pred.tif')
        print(pred.shape)
        scipy.misc.imsave(fnpred, pred.reshape(c.height, c.width) * np.float32(255))

        mod = math.floor(i / num_images * 10) 
        if mod > premod and mod > 0:
            print(str(mod*10)+'%..',end="",flush=True)
        premod = mod

    if not probs_only: submission.close()
    print("Prediction took {:.3f}s".format(time.clock() - start_time))





def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", dest="version",  help="version", default=0.0)
    parser.add_argument("-d", "--depth", dest="depth",  help="depth", default=6)
    parser.add_argument("-f", "--filters", dest="filters",  help="filters", default=8)
    parser.add_argument("-resize", dest="resize",  help="resize", default=0.0) 
    parser.add_argument("-cv", dest="cv",  help="cv", default=0) 
    parser.add_argument("-cvi", dest="cvi",  help="cvi", default=0) 
    parser.add_argument("-ms", "-minsize", dest="minsize",  help="minsize", default=5000) 
    parser.add_argument("-tta", "-tta", dest="tta",  help="tta", default=1) 
    parser.add_argument("-seed", dest="seed",  help="seed", default=1234)
    options = parser.parse_args()

    version = float(options.version)
    depth = int(options.depth)
    filters = int(options.filters)
    resize = float(options.resize)
    cv = int(options.cv)
    cvi = int(options.cvi)
    minsize = int(options.minsize)
    tta = int(options.tta)
    seed = int(options.seed)

    print("Arguments:")
    print("----------")
    print(options)
    print()

    if cv<=0 and cvi==0:
        test_model(version, depth, filters, resize=resize, minsize=minsize, tta_num=tta, seed=seed)
    elif cvi>0:
        test_model(version, depth, filters, resize=resize, probs_only=True, cv=cvi, tta_num=tta, seed=seed)
    else:
        for fold in range(1,cv+1):
            print("cv fold "+str(fold)+"\n")
            test_model(version, depth, filters, resize=resize, probs_only=True, cv=fold, tta_num=tta, seed=seed)
        join_models_cv(version, minsize=minsize)

if __name__ == '__main__':
    main()
