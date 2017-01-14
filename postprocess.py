import misc
import test
import glob
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter


def dice_predict(pred, tgt):
    predeq = (pred >= 0.5)
    tgteq = (tgt >= 0.5)
    den = predeq.sum() + tgteq.sum()
    if den == 0: return -1
    return -2* (predeq*tgteq).sum()/den


version = '15.300203201'
datadir='train'
showdir='show/v'+version
minsize=5000
mask_names = glob.glob(datadir+'/*_mask.tif')
misc.sort_nicely(mask_names)



dice=0
num=0
for i in range(len(mask_names)):
    mask_name = mask_names[i]
    mask = misc.load_image(mask_name)
    if mask.sum() <= minsize:
        continue
    pred_name = mask_name.replace(datadir, showdir).replace("_mask", "")
    predthr = misc.load_image(pred_name)
    dice += dice_predict(mask,predthr)
    num += 1

dice/=num
print("original dice: "+str(dice))


iterations=2
# for sigma in range(6,11):
for sigma in range(1,6):
    dice=0
    num=0
    for i in range(len(mask_names)):
        mask_name = mask_names[i]
        mask = misc.load_image(mask_name)
        if mask.sum() <= minsize:
            continue
        pred_name = mask_name.replace(datadir, showdir).replace("_mask", "")
        predthr = misc.load_image(pred_name)
        predext = gaussian_filter(binary_dilation(predthr, iterations=iterations).astype(predthr.dtype), sigma=sigma)
        predthr = ( ((predext>0.5)+predthr)>0 ).astype(predthr.dtype)
        dice += dice_predict(mask,predthr)
        num += 1
    #
    dice/=num
    print("iterations: "+str(iterations)+" sigma: "+str("sigma ")+" dice: "+str(dice))




import config as c
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

import img_augmentation as aug

img=misc.load_image('train/1_1.tif')
shape = img.shape[1:]



tform_centering = aug.build_centering_transform(img.shape[1:], None)
tform_center, tform_uncenter = aug.build_center_uncenter_transforms(img.shape[1:])
tform_augment = aug.random_perturbation_transform(aug_params)
tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
img = aug.fast_warp(img, tform_centering + tform_augment, output_shape=None, mode='constant')

t=tform_centering + tform_augment
t.params=np.linalg.inv(t.params)
nimg = aug.fast_warp(img, t, output_shape=None, mode='constant')

nimg = skimage.transform.warp(img[0], (tform_centering + tform_augment).inverse, output_shape=img.shape[1:])

