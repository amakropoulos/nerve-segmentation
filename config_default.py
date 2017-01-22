modelversion = 1
depth = 6
filters = 8
filter_size = 3
regularization = 0
resize = 0
learning_rate = [3e-3, 3e-3]

autoencoder = False
autoencoder_dropout = 0.3
pretrain = None

train_loss = 'dice'
val_loss = 'dice'

height = 420
width = 580
batch_size = 25
num_epochs = 100

augment = True
elastic = False
alpha = 100  # params of deformation-based aug
sigma = 4   # same
elastic_warps_dir = ''  # precomputed warps if specified

non_elastic = True
shift = 10;
rotation = 10;
shear = 2;
scale = 0;
flip = False
stretch = True
