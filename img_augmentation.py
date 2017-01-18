import skimage
import skimage.transform
from skimage.transform._warps_cy import _warp_fast
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import pickle
import glob
import gzip

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}

def fast_warp(img, tf, output_shape=None, mode='constant', order=0):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params
    if output_shape is None:
        t_img = np.zeros_like(img);
    else:
        t_img = np.zeros((img.shape[0],) + output_shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, output_shape=output_shape, 
                              mode=mode, order=order)
    return t_img


def build_centering_transform(image_shape, target_shape):
    rows, cols = image_shape
    if target_shape is None:
        trows,tcols = image_shape
    else:
        trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False): 
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(augmentation_params, rng=np.random):
    zoom_range = augmentation_params["zoom_range"]
    rotation_range = augmentation_params["rotation_range"]
    shear_range = augmentation_params["shear_range"]
    translation_range = augmentation_params["translation_range"]
    do_flip = augmentation_params["do_flip"]
    allow_stretch = augmentation_params["allow_stretch"]

    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0) # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

# do the augmentation
# if rng is None, it can be used for test-time augmentation
def perturb(img, label, augmentation_params, target_shape=None, rng=np.random):
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = random_perturbation_transform(augmentation_params, rng=rng)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    img = fast_warp(img, tform_centering + tform_augment, output_shape=target_shape, mode='constant')
    label = fast_warp(label, tform_centering + tform_augment, output_shape=target_shape, mode='constant')
    return [img, label]

# augmentation at test time
def test_time_augmentation(img, predict_model, num, shape, augmentation_params):
    tform_centering = build_centering_transform(img.shape[2:], None)
    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape[2:])

    images = np.zeros((num, 1, img.shape[2], img.shape[3]), dtype='float32')
    augments = []
    for i in range(num):
        if i == 0:
            images[i] = img
            augments.append(None)
        else:
            tform_augment = random_perturbation_transform(augmentation_params)
            tform_augment = tform_centering + tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
            images[i][0] = fast_warp(img[0], tform_augment, output_shape=None, mode='constant')
            augments.append(tform_augment)

    preds = predict_model(images)
    pred = np.zeros(shape, dtype='float32')
    for i in range(num):
        if i == 0:
            cur_pred = preds[i]
        else:
            tform_augment = augments[i]
            tform_augment.params=np.linalg.inv(tform_augment.params)
            cur_pred = fast_warp(preds[i], tform_augment, output_shape=None, mode='constant')

        if shape != img.shape:
            cur_pred = cv2.resize(cur_pred[0], (shape[3], shape[2]), interpolation=cv2.INTER_LINEAR)
            cur_pred = cur_pred.reshape( (shape[1],shape[2],shape[3]) )
        pred[0] += cur_pred

    pred/=num
    return pred


def augment(image, label, augmentation_params):
    if augmentation_params['elastic']:
        [ image, label ] = elastic_transform(img, lbl, augmentation_params)
    if augmentation_params['non_elastic']:
        [ image, label ] = perturb(img, lbl, augmentation_params)
    return [image, label]

# elastic deformation
# if an elastic_warps_dir is provided at the augmentation_params it will use one of the warps stored there
elastic_dir_contents = []
def elastic_transform(image, label, augmentation_params, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[1:];
    elastic_warps_dir = augmentation_params['elastic_warps_dir']
    if elastic_warps_dir != "":
        global elastic_dir_contents
        if len(elastic_dir_contents) == 0:
            elastic_dir_contents = glob.glob(elastic_warps_dir+'/*pklz')
        selection = random_state.randint(0, len(elastic_dir_contents)-1)
        f = gzip.open(elastic_dir_contents[selection],'rb')
        indices = pickle.load(f)
        f.close()
    else:
        alpha = augmentation_params["alpha"]
        sigma = augmentation_params["sigma"]
        # alpha = random_state.uniform(alpha)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    #return map_coordinates(image, indices, order=1).reshape(shape)
    resimage = np.zeros_like(image);
    reslabel = np.zeros_like(label);
    for i in range(image.shape[0]):
        resimage[i] = map_coordinates(image[i], indices, order=1).reshape(shape)
        reslabel[i] = map_coordinates(label[i], indices, order=1, mode='nearest').reshape(shape)
    return [resimage, reslabel];


# generate elastic warps in the elastic_warps_dir
def generate_elastic_warps(times, alpha, sigma, elastic_warps_dir):    
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


######################## OTHER STUFF NOT CURRENTLY NEEDED ########################


def perturb_fixed(img, label, tform_augment, target_shape=None):
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    img = fast_warp(img, tform_centering + tform_augment, output_shape=target_shape, mode='constant')
    label = fast_warp(label, tform_centering + tform_augment, output_shape=target_shape, mode='constant')
    return [img, label]

# fancy PCA
U = np.array([[-0.60,0.55,0.58],
    [-0.60,0.17,-0.77],
    [-0.53,-0.82,0.23]],dtype = np.float32);
EV = np.array([1.1,0.16,0.067],dtype = np.float32);

def augment_color(img, sigma=0.3):
    if sigma<=0.0:
        return img;
    color_vec = np.random.normal(0.0, sigma, 3)
    alpha = color_vec.astype(np.float32) * EV
    noise = np.dot(U, alpha.T)
    return img + noise[:, np.newaxis, np.newaxis]

    

def load_augment(fname, w, h, aug_params=no_augmentation_params,
                 transform=None, sigma=0.0, color_vec=None):
    """Load augmented image with output shape (w, h).

    Default arguments return non augmented image of shape (w, h).
    To apply a fixed transform (color augmentation) specify transform
    (color_vec). 
    To generate a random augmentation specify aug_params and sigma.
    """
    img = load_image(fname)
    if transform is None:
        img = perturb(img, augmentation_params=aug_params, target_shape=(w, h))
    else:
        img = perturb_fixed(img, tform_augment=transform, target_shape=(w, h))

    np.subtract(img, MEAN[:, np.newaxis, np.newaxis], out=img)
    np.divide(img, STD[:, np.newaxis, np.newaxis], out=img)
    img = augment_color(img, sigma=sigma, color_vec=color_vec)
    return img


def match_pretrained(img,S): #fill with 0s to match pretrained network size
    imgs_resize = np.zeros((img.shape[0],S,S),dtype=np.float32)
    w,h = img.shape[1:];
    a,b = (S-w)/2,(S-h)/2;
    imgs_resize[:,a:a+w,b:b+h] = img;
    return imgs_resize;
