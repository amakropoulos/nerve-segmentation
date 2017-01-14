import scipy.misc
import numpy as np
import glob
import config as c
import pickle
import os.path
import re
import math
import img_augmentation as aug
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import gaussian_filter
import gzip
import skimage
import cv2

def save_image(filename, img):
    scipy.misc.imsave(filename, img.reshape(c.height, c.width))

def load_image(filename):
    img = scipy.misc.imread(filename)/ np.float32(255)
    return img.transpose(0,1).reshape(1, c.height, c.width)


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



def dt(image, sigma = 5):
    img = image.reshape(c.height, c.width)
    return gaussian_filter(img, sigma).reshape(1, c.height, c.width)


def load_data(val_pct = 0.1, cv=0, dtbased=False, seed = 1234, first=0, datadir='train', only_names=True):
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

    if cv > 0:
        sind = num_subjects[0] * (cv-1)
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
                image = load_image(image_name, std=True)
                mask = load_image(mask_name)
                if dtbased: mask = dt(mask)
                Xs[d][ind] = image
                ys[d][ind] = mask
                ind = ind + 1

    return Xs[1], ys[1], Xs[0], ys[0]



def augment_image(image, label):
    aug_params = {
        'zoom_range': (1/(1+c.scale), 1+c.scale),
        'rotation_range': (-c.rotation, c.rotation),
        'shear_range': (-c.shear, c.shear),
        'translation_range': (-c.shift, c.shift),
        'do_flip': c.flip,
        'allow_stretch': c.stretch,
        'elastic_warps_dir':c.elastic_warps_dir,
        'alpha': c.alpha,
        'sigma': c.sigma,
    }
    if c.elastic:
        [ image, label ] = aug.elastic_transform(image, label, aug_params)
    if c.non_elastic:
        [ image, label ] = aug.perturb(image, label, aug_params)
    # elif c.data_based:
    #     [ image, label ] = augment_image_data_based(image, label, c.warps_dir)

    return [ image, label] 


def tta(img, predict_model):
    aug_params = {
        'zoom_range': (1/(1+c.scale), 1+c.scale),
        'rotation_range': (-c.rotation, c.rotation),
        'shear_range': (-c.shear, c.shear),
        'translation_range': (-c.shift, c.shift),
        'do_flip': c.flip,
        'allow_stretch': c.stretch,
        'elastic_warps_dir':c.elastic_warps_dir,
        'alpha': c.alpha,
        'sigma': c.sigma,
    }
    tform_centering = aug.build_centering_transform(img.shape[2:], None)
    tform_center, tform_uncenter = aug.build_center_uncenter_transforms(img.shape[2:])
    tform_augment = aug.random_perturbation_transform(aug_params)
    tform_augment = tform_centering + tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    nimg = img+0
    nimg[0] = aug.fast_warp(img[0], tform_augment, output_shape=None, mode='constant')
    pred = predict_model(nimg)
    tform_augment.params=np.linalg.inv(tform_augment.params)
    nimg[0] = aug.fast_warp(pred[0], tform_augment, output_shape=None, mode='constant')
    return nimg


def tta2(img, predict_model, num, shape):
    aug_params = {
        'zoom_range': (1/(1+c.scale), 1+c.scale),
        'rotation_range': (-c.rotation, c.rotation),
        'shear_range': (-c.shear, c.shear),
        'translation_range': (-c.shift, c.shift),
        'do_flip': c.flip,
        'allow_stretch': c.stretch,
        'elastic_warps_dir':c.elastic_warps_dir,
        'alpha': c.alpha,
        'sigma': c.sigma,
    }
    tform_centering = aug.build_centering_transform(img.shape[2:], None)
    tform_center, tform_uncenter = aug.build_center_uncenter_transforms(img.shape[2:])

    images = np.zeros((num, 1, img.shape[2], img.shape[3]), dtype='float32')
    augments = []
    for i in range(num):
        if i == 0:
            images[i] = img
            augments.append(None)
        else:
            tform_augment = aug.random_perturbation_transform(aug_params)
            tform_augment = tform_centering + tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
            images[i][0] = aug.fast_warp(img[0], tform_augment, output_shape=None, mode='constant')
            augments.append(tform_augment)

    preds = predict_model(images)
    pred = np.zeros(shape, dtype='float32')
    for i in range(num):
        if i == 0:
            cur_pred = preds[i]
        else:
            tform_augment = augments[i]
            tform_augment.params=np.linalg.inv(tform_augment.params)
            cur_pred = aug.fast_warp(preds[i], tform_augment, output_shape=None, mode='constant')

        if shape != img.shape:
            cur_pred = cv2.resize(cur_pred[0], (shape[3], shape[2]), interpolation=cv2.INTER_LINEAR)
            cur_pred = cur_pred.reshape( (shape[1],shape[2],shape[3]) )
        pred[0] += cur_pred

    pred/=num
    return pred


# import cv2
# warp_dir_contents = []
# def augment_image_data_based(image, label, warpdir='warps/pass'):
#     global warp_dir_contents
#     if len(warp_dir_contents) == 0:
#         warp_dir_contents = glob.glob(warpdir+'/*.pickle')
#     selection = np.random.randint(0, len(warp_dir_contents))
#     with open(warp_dir_contents[selection], 'rb') as re:
#         [warp_matrix, cc, overlap] = pickle.load(re)
#     image[0] = np.transpose( cv2.warpAffine( np.transpose(image[0]), warp_matrix, image[0].shape, flags=cv2.INTER_LINEAR) )
#     label[0] = np.transpose( cv2.warpAffine( np.transpose(label[0]), warp_matrix, label[0].shape, flags=cv2.INTER_NEAREST) )
#     return image, label


def generate_elastic_warps(times, alpha=c.alpha, sigma=c.sigma, elastic_warps_dir=c.elastic_warps_dir):    
    if not os.path.exists(elastic_warps_dir):
        os.makedirs(elastic_warps_dir)
        next = 0
    else:
        elastic_dir_contents = glob.glob(elastic_warps_dir+'/*pklz')
        next = len(elastic_dir_contents)
    
    for i in range(next, next+times):
        alpha = np.random.uniform(alpha)
        shape = (c.height, c.width)
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)).astype(np.float32)

        fn = elastic_warps_dir+"/"+str(i)+".pklz"
        f = gzip.open(fn,'wb')
        pickle.dump(indices,f)
        f.close()

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



def get_params_dir(version, autoencoder=False, cv=0, seed=1234):
    suffix = "" if not autoencoder else "-autoencoder"
    suffix = suffix 
    if cv > 0:
        suffix ="{}/cv{}".format(suffix, cv)
    else:
        suffix ="{}/seed{}".format(suffix, seed)
    return os.path.join(c.params_dir, 'v{}{}'.format(version, suffix))


def completed(version, cv=0, seed=1234):
    epoch = -1
    batch = 0
    fn = get_params_dir(version, cv=cv, seed=seed)+'/checkpoint.pickle'
    if os.path.isfile(fn):
        with open(fn, 'rb') as re:
            [param_vals, epoch, batch] = pickle.load(re)
    return [epoch, batch]


def resume(model, version, autoencoder=False, cv=0, seed=1234):
    epoch = -1
    batch = 0
    fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/checkpoint.pickle'
    if os.path.isfile(fn):
        with open(fn, 'rb') as re:
            [param_vals, epoch, batch] = pickle.load(re)
            import lasagne
            lasagne.layers.set_all_param_values(model, param_vals)
    else:
        epoch = load_last_params(model, version) + 1
    [mve, train_error, val_error, val_accuracy] = load_results(version, cv=cv)
    return [epoch, batch, mve, train_error, val_error, val_accuracy]


def save_progress(model, version, epoch, batch, autoencoder=False, cv=0, seed=1234):
    suffix = "" if not autoencoder else "-autoencoder"
    fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/checkpoint.pickle'
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    import lasagne
    param_vals = lasagne.layers.get_all_param_values(model)
    stuff = [param_vals, epoch, batch]
    with open(fn, 'wb') as wr:
        pickle.dump(stuff, wr)


def load_results(version, autoencoder=False, cv=0, seed=1234):
    mve = None
    train_error = {} 
    val_error = {}
    val_accuracy = {}
    fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/results.pickle'
    if os.path.isfile(fn):
        with open(fn, 'rb') as re:
            [mve, train_error, val_error, val_accuracy] = pickle.load(re)
    return [mve, train_error, val_error, val_accuracy]


def save_results(version, mve, train_error, val_error, val_accuracy, autoencoder=False, cv=0, seed=1234):
    fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/results.pickle'
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    with open(fn, 'wb') as wr:
        pickle.dump([mve, train_error, val_error, val_accuracy], wr)


def load_last_params(model, version, best=False, autoencoder=False, cv=0, seed=1234):
    fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/params_e*.npz'
    if best:
        fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/params_best_e*.npz'
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


def save_params(model, version, epoch, best=False, autoencoder=False, cv=0, seed=1234):
    fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/params_e{}.npz'.format(epoch)
    if best:
        fn = get_params_dir(version, autoencoder=autoencoder, cv=cv, seed=seed)+'/params_best_e{}.npz'.format(epoch) 
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    import lasagne
    param_vals = lasagne.layers.get_all_param_values(model)
    np.savez(fn, *param_vals)


def stats(datadir="train"):
	from skimage import measure
	X, y, X_val, y_val = load_data(0, datadir=datadir, only_names=True)
	maxpos = [-1, -1]
	minpos = [c.height, c.width]
	meanpos = [0, 0]
	maxsz = [-1, -1]
	minsz = [c.height, c.width]
	meansz = [0, 0]
	maxlccs = -1
	meanlccs = 0
	positive = 0
	negative = 0
	szs=[]

	for p in range(len(y)):
	    mask = load_image(y[p])
	    zi, zj = mask.reshape(c.height,c.width).nonzero()
	    if len(zi):
	        szs.append(len(zi))
	        i, j = min(zi), min(zj)
	        maxpos = [max(maxpos[0],i), max(maxpos[1],j)]
	        minpos = [min(minpos[0],i), min(minpos[1],j)]
	        meanpos[0] += i
	        meanpos[1] += j
	        i, j = max(zi)-min(zi), max(zj)-min(zj)
	        maxsz = [max(maxsz[0],i), max(maxsz[1],j)]
	        minsz = [min(minsz[0],i), min(minsz[1],j)]
	        meansz[0] += i
	        meansz[1] += j
	        positive += 1
	        lccs = np.max(measure.label(mask))
	        maxlccs = max(lccs, maxlccs)
	        meanlccs += lccs
	    else:
	        negative+=1
	szs.sort()
	meanpos[0] = int(round(meanpos[0] / positive))
	meanpos[1] = int(round(meanpos[1] / positive))
	meansz[0] = int(round(meansz[0] / positive))
	meansz[1] = int(round(meansz[1] / positive))
	meanlccs/=positive
	print("positive: "+str(positive)+" negative:"+str(negative))
	txt="position avg: ("+str(meanpos[0])+","+str(meanpos[1])+")"
	txt+="  min: ("+str(minpos[0])+","+str(minpos[1])+")"
	txt+="  max: ("+str(maxpos[0])+","+str(maxpos[1])+")"
	print(txt)
	txt="size avg: ("+str(meansz[0])+","+str(meansz[1])+")"
	txt+="  min: ("+str(minsz[0])+","+str(minsz[1])+")"
	txt+="  max: ("+str(maxsz[0])+","+str(maxsz[1])+")"
	print(txt)
	txt="lccs max: "+str(maxlccs)+" mean:"+str(meanlccs)
	print(txt)
	txt="size min: "+str(szs[0])+" max: "+str(szs[len(szs)-1])+" 5%:"+str(szs[round(0.5*len(szs))])+" 95%:"+str(szs[round(0.95*len(szs))])+ " median:"+str(szs[round(0.5*len(szs))])
	print(txt)


def spatial_prob(datadir="train"):
    from skimage import measure
    X, y, X_val, y_val = load_data(0, datadir=datadir, only_names=True)
    prob = np.zeros((1, c.height, c.width), dtype='float32')

    num = 0
    for p in range(len(y)):
        mask = load_image(y[p]).astype(np.float32)
        if mask.sum():
            prob += mask
            num += 1

    prob /= num
    scipy.misc.imsave("spatial_prob.tiff", prob.reshape(c.height, c.width) * np.float32(255))





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



def find_scaling(depth=5):
    # depth=5
    orig_shape=np.array([c.width, c.height])
    for factor in np.linspace(0,1,1001):
        if factor==0: continue
        factor_shape = orig_shape * factor
        shape = factor_shape
        for i in range(depth-1): 
            shape = np.floor(shape/2)
        for i in range(depth-1): 
            shape = shape*2
        shape_diff = factor_shape - shape
        if (shape_diff%2).sum() == 0 and ( shape_diff[0]/2 == shape_diff[1]/2 ):
            print(str(factor)+"   "+str(shape_diff[0]) )






# def augment_data(times, datadir="train", augdir="train-augmented", elastic = True):
#     if not os.path.exists(augdir):
#         os.makedirs(augdir)

#     aug_params = {
#         'zoom_range': (1/(1+c.scale), 1+c.scale),
#         'rotation_range': (-c.rotation, c.rotation),
#         'shear_range': (-c.shear, c.shear),
#         'translation_range': (-c.shift, c.shift),
#         'do_flip': c.flip,
#         'allow_stretch': c.stretch,
#         'alpha': c.alpha,
#         'sigma': c.sigma,
#     }

#     mask_names = glob.glob(datadir+'/*_mask.tif')
#     sort_nicely(mask_names)

#     for mask_name in mask_names:
#         image_name = mask_name.replace("_mask", "")
#         image = load_image(image_name)
#         mask = load_image(mask_name)

#         s = mask_name.split("/")[1].split("_")[0]
#         image_num = os.path.basename(image_name).split("_")[-1].split('.')[0]   

#         fn = augdir+'/'+s+'_'+str(image_num)+'-a*_mask.tif'
#         aug_ex_images = glob.glob( fn )
#         num_ex = len(aug_ex_images)

#         print(image_name)
#         for i in range(times):
#             print(i)
#             if elastic:
#                 [ image_aug, mask_aug ] = aug.elastic_transform(image, mask, aug_params)
#             else:
#                 [ image_aug, mask_aug ] = aug.perturb(image, mask, aug_params)

#             imgname = augdir+'/'+s+'_'+str(image_num)+'-a'+str(num_ex+i+1)+'.tif' 
#             maskname = augdir+'/'+s+'_'+str(image_num)+'-a'+str(num_ex+i+1)+'_mask.tif' 
#             scipy.misc.imsave(imgname, image_aug.reshape(c.height,c.width))
#             scipy.misc.imsave(maskname, mask_aug.reshape(c.height,c.width))

#     # return [Xa, ya]
