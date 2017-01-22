import scipy.misc
import numpy as np
import glob
import config_default as c
import pickle
import os.path
import re
import math
import img_augmentation as aug
import gzip
#import skimage
import cv2


params_dir = "params"

######################## LOAD/SAVE IMAGE METHODS ########################

def save_image(filename, img):
    scipy.misc.imsave(filename, img.reshape(c.height, c.width))

def load_image(filename):
    img = scipy.misc.imread(filename)/ np.float32(255)
    return img.transpose(0,1).reshape(1, c.height, c.width)

# load all images/labels from the data_dir and randomly split them into train/val
# if only_names==True then only the names of the files are provided, 
# otherwise the images are loaded as well
def load_data(val_pct = 0.1, fold=0, seed = 1234, first=0, datadir='train', only_names=True):
    mask_names = glob.glob(datadir+'/*_mask.tif')
    subjects = uniq([i.split("/")[1].split("_")[0] for i in mask_names])
    if first>0:
        subjects = subjects[0:first]
    sort_nicely(subjects)

    num_subjects = {}
    # validation subjects
    num_subjects[0] = math.ceil(val_pct*len(subjects))
    # train subjects
    num_subjects[1] = len(subjects) - num_subjects[0]

    if fold > 0:
        sind = num_subjects[0] * (fold-1)
        lind = sind + num_subjects[0] 
        if lind > len(subjects):
            sub = lind - len(subjects)
            sind-=sub
            lind-=sub

        subjects = np.hstack([subjects[sind:lind], subjects[0:sind], subjects[lind:]]).tolist()
    else:
        np.random.seed(seed)
        np.random.shuffle(subjects)

    Xs = {}
    ys = {}
    for d in range(2):
        d_num_subjects = num_subjects[d]
        if d_num_subjects == 0:
            Xs[d] = None
            ys[d] = None
            continue;

        mask_names = [];
        for i in range(d_num_subjects):
            s = subjects.pop(0)
            mask_names = mask_names + glob.glob(datadir+'/'+s+'_*_mask.tif')

        num_images = len(mask_names) 
        if d==1:
            np.random.shuffle(mask_names)
        else:
            sort_nicely(mask_names)
        if only_names:
            Xs[d] = {}
            ys[d] = {}
            ind=0
            for mask_name in mask_names:
                Xs[d][ind] = mask_name.replace("_mask", "")
                ys[d][ind] = mask_name
                ind = ind + 1
        else:
            Xs[d] = np.zeros((num_images, 1, c.height, c.width), dtype='float32')
            ys[d] = np.zeros((num_images, 1, c.height, c.width), dtype='float32')
            ind=0
            for mask_name in mask_names:
                image_name = mask_name.replace("_mask", "")
                image = load_image(image_name)
                mask = load_image(mask_name)
                Xs[d][ind] = image
                ys[d][ind] = mask
                ind = ind + 1

    return Xs[1], ys[1], Xs[0], ys[0]



######################## USEFUL METHODS ########################

def uniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  return output

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    l.sort(key=alphanum_key)


######################## LOAD/SAVE RESULTS METHODS ########################


def get_params_dir(version, fold=0, seed=1234):
    suffix=""
    if fold > 0:
        suffix ="/fold{}".format(fold)
    else:
        suffix ="/seed{}".format(seed)
    return os.path.join(params_dir, '{}{}'.format(version, suffix))


# load config file
def load_config(version):
    # load default config
    global c
    import config_default as c

    # merge default and model config
    model_config = params_dir+"/"+str(version)+"/config.py"
    if os.path.exists(model_config):
        import importlib
        import sys
        sys.path.append(os.path.dirname(model_config))
        mname = os.path.splitext(os.path.basename(model_config))[0]
        c2 = importlib.import_module(mname)
        c.__dict__.update(c2.__dict__)
    else:
        import warnings
        warnings.warn("using default parameters")

    # params for augmentation
    c.aug_params = {
        'use': c.augment,
        'non_elastic': c.non_elastic,
        'zoom_range': (1/(1+c.scale), 1+c.scale),
        'rotation_range': (-c.rotation, c.rotation),
        'shear_range': (-c.shear, c.shear),
        'translation_range': (-c.shift, c.shift),
        'do_flip': c.flip,
        'allow_stretch': c.stretch,
        'elastic': c.elastic,
        'elastic_warps_dir':c.elastic_warps_dir,
        'alpha': c.alpha,
        'sigma': c.sigma,
    }
    return c

def resume(version=0, model = None,  fold=0, seed=1234):
    epoch = -1
    batch = 0
    fn = get_params_dir(version, fold=fold, seed=seed)+'/checkpoint.pickle'
    if os.path.isfile(fn):
        with open(fn, 'rb') as re:
            [param_vals, epoch, batch] = pickle.load(re)
            if model is not None:
                import lasagne
                lasagne.layers.set_all_param_values(model, param_vals)
    else:
        epoch = load_last_params(model, version) + 1
    [mve, train_error, val_error, val_accuracy] = load_results(version, fold=fold)
    return [epoch, batch, mve, train_error, val_error, val_accuracy]


def save_progress(model, version, epoch, batch, fold=0, seed=1234):
    fn = get_params_dir(version, fold=fold, seed=seed)+'/checkpoint.pickle'
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    import lasagne
    param_vals = lasagne.layers.get_all_param_values(model)
    stuff = [param_vals, epoch, batch]
    with open(fn, 'wb') as wr:
        pickle.dump(stuff, wr)


def load_results(version, fold=0, seed=1234):
    mve = None
    train_error = {} 
    val_error = {}
    val_accuracy = {}
    fn = get_params_dir(version, fold=fold, seed=seed)+'/results.pickle'
    if os.path.isfile(fn):
        with open(fn, 'rb') as re:
            [mve, train_error, val_error, val_accuracy] = pickle.load(re)
    return [mve, train_error, val_error, val_accuracy]


def save_results(version, mve, train_error, val_error, val_accuracy, fold=0, seed=1234):
    fn = get_params_dir(version, fold=fold, seed=seed)+'/results.pickle'
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    with open(fn, 'wb') as wr:
        pickle.dump([mve, train_error, val_error, val_accuracy], wr)


def load_last_params(model, version, best=False, fold=0, seed=1234):
    fn = get_params_dir(version, fold=fold, seed=seed)+'/params_e*.npz'
    if best:
        fn = get_params_dir(version, fold=fold, seed=seed)+'/params_best_e*.npz'
    param_names = glob.glob( fn )
    if len(param_names) == 0:
        return -1
    sort_nicely(param_names)
    paramfile = param_names[-1]
    load_params(model, paramfile)
    epoch = os.path.basename(paramfile).split("_e")[-1].split('.')[0]
    return tryint(epoch)


def load_params(model, fn):
    with np.load(fn) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    import lasagne
    lasagne.layers.set_all_param_values(model, param_values)


def save_params(model, version, epoch, best=False, fold=0, seed=1234):
    fn = get_params_dir(version, fold=fold, seed=seed)+'/params_e{}.npz'.format(epoch)
    if best:
        fn = get_params_dir(version, fold=fold, seed=seed)+'/params_best_e{}.npz'.format(epoch) 
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    import lasagne
    param_vals = lasagne.layers.get_all_param_values(model)
    np.savez(fn, *param_vals)



def completed(version, fold=0, seed=1234):
    completed = get_params_dir(version, fold=fold, seed=seed)+'/completed'
    if os.path.isfile(completed):
        with open(completed, 'r') as rf:
            print("best score: "+rf.readline())
        return True
    return False

def finish(version, mve, fold=0, seed=1234):
    completed = get_params_dir(version, fold=fold, seed=seed)+'/completed'
    with open(completed, 'w') as wf:
         wf.write(str(mve))


######################## OTHER STUFF NOT CURRENTLY NEEDED ########################



def run_length(img):
    # img is binary mask image, shape (r,c)
    bytes = img.reshape(c.height*c.width, order='F')
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for cb in bytes:
        if ( cb == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    z = ''
    for rr in runs:
        z+='{} {} '.format(rr[0],rr[1])
    return z[:-1]



# trainmean = None
# trainstd = None
# def load_image(filename, std=False):
#     img = (scipy.misc.imread(filename)/ np.float32(255)).transpose(0,1).reshape(1, c.height, c.width)
#     if std:
#         global trainmean, trainstd
#         if trainmean is None:
#             with open('meanstd.pickle', 'rb') as re:
#                 [trainmean, trainstd] = pickle.load(re)
#         img = (img - trainmean) / trainstd
#     return img

# def preprocess(datadir='train'):
#     mask_names = glob.glob(datadir+'/*_mask.tif')
#     X = np.zeros((len(mask_names), 1, c.height, c.width), dtype='float32')
#     i = 0
#     for mask_name in mask_names:
#         image_name = mask_name.replace("_mask", "")
#         X[i] = load_image(filename)
#         i += 1
#     trainmean = np.mean(X)
#     trainstd = np.std(X)
#     with open('meanstd.pickle', 'wb') as wr:
#         pickle.dump([trainmean, trainstd], wr)
