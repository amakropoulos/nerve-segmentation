# Images
# =====================================
# Image extension
image_ext = '.tif'
# Image height
height = 420
# Image width
width = 580

# Training
# =====================================

# model parameters (for network() of model.py):
# model version 
modelversion = 3
# network depth
depth = 6
# network number of filters
filters = 8
# network filter size (convolution)
filter_size = 3
# learn autoencoder (True) or segmentation (False)
autoencoder = False
# dropout of input layer for autoencoder
autoencoder_dropout = 0.3

# training parameters:
# regularization factor (l2 regularization at each layer)
regularization = 0
# image resize factor
resize = 0
# starting/ending learning rate (this is linearly divided among the epochs)
learning_rate = [3e-3, 3e-3]
# pretrain model version
pretrain = None
# train loss function
train_loss = 'dice'
# validation loss function
val_loss = 'dice'
# number of images used per batch
batch_size = 25
# number of epochs
num_epochs = 100

# augmentation parameters:
# use augmentation
augment = True
# do elastic augmentation
elastic = False
# magnitude of deformation-based augmentation
alpha = 200  
# sigma of deformation-based augmentation
sigma = 20  
# use precomputed warps from this directory if specified
elastic_warps_dir = ''
# do non-elastic augmentation
non_elastic = True
# max image shift
shift = 10;
# max image rotation
rotation = 10;
# max image shear
shear = 2;
# max image scale
scale = 0;
# image flipping
flip = False
# image stretching
stretch = True
