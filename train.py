import os
import sys
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import time
import math
import lasagne 
import argparse
import cv2
import model
import misc
import test
import loss as L
import img_augmentation as aug
# import scipy.misc
# from skimage import measure
# import glob

# print available GPU memory
def print_GPU_memory():
    import theano.sandbox.cuda.basic_ops as sbcuda
    GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
    print("GPUFreeMemory: "+str(GPUFreeMemoryInBytes))
    print("")


# add regularization
def add_regularization(network):
    import lasagne.regularization as reg
    layers = lasagne.laeyrs.get_all_layers(network)
    penalty_layers = []
    for l in layers:
        if isinstance(l, lasagne.layers.Conv2DLayer):
            penalty_layers.append(l)
    return reg.regularize_layer_params(penalty_layers, reg.l2)


# load batch and apply augmentation
def getbatch(inputs, targets, shape, batchsize, aug_params, shuffle=False, seed=-1, init_batch=0, augment=False, resize=0):
    indices = np.arange(len(inputs))
    if shuffle:
        if seed>=0: 
            np.random.seed(seed)
        np.random.shuffle(indices)

    Xs = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    ys = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    d=0
    for ii in range(init_batch*batchsize, len(inputs)):
        i = ii
        if shuffle:
            i = indices[ii]
        img = misc.load_image(inputs[i])
        lbl = misc.load_image(targets[i])
        if aug_params["use"]:
            [ image, label ] = aug.augment(img, lbl, aug_params)

        if resize:
            img = cv2.resize(img[0], (shape[3], shape[2]), interpolation=cv2.INTER_CUBIC).reshape(shape[1:])
            lbl = cv2.resize(lbl[0], (shape[3], shape[2]), interpolation=cv2.INTER_NEAREST).reshape(shape[1:])
        Xs[d] = img
        ys[d] = lbl
        d+=1
        if d == batchsize:
            d = 0
            yield Xs, ys
    if(d>0):
        yield Xs[0:d,:,:,:], ys[0:d,:,:,:]

# train the model
def train_model(version=0, train_dir = 'train', fold=1, num_folds=10, seed=1234):
    # completed?
    if misc.completed(version, fold, seed): return;

    # load model config 
    c = misc.load_config(version)

    # define image shape (original or resized)
    batch_size = c.batch_size
    if c.resize != 0:
        shape = (None, 1, round(c.height*c.resize), round(c.width*c.resize) ) 
    else:
        shape = (None, 1, c.height, c.width)

    # load data
    print("Loading data..")
    start_time = time.clock()   
    X_train, y_train, X_val, y_val = misc.load_data(val_pct=1/num_folds, datadir=train_dir, fold=fold, seed=seed)
    print("took " + str(time.clock()-start_time )+"s")

    # build network
    input_var = T.tensor4('input')
    label_var = T.tensor4('label')
    net = model.network(input_var, shape, filter_size=c.filter_size, version=c.modelversion, depth=c.depth, num_filters=c.filters, autoencoder=c.autoencoder, autoencoder_dropout=c.autoencoder_dropout)
    output = lasagne.layers.get_output(net['output'])
    output_det = lasagne.layers.get_output(net['output'], deterministic=True)

    # loss function, regularization etc
    train_loss_function = getattr(L, c.train_loss)
    val_loss_function = getattr(L, c.val_loss)
    train_loss = train_loss_function(output, label_var)
    val_loss = val_loss_function(output_det, label_var)
    if c.regularization>0:
         train_loss = train_loss + c.regularization*add_regularization(net['output'])
    lr = theano.shared(lasagne.utils.floatX(c.learning_rate[0]))
    # learning rate can be epoch-specific
    lr_epoch = np.linspace(c.learning_rate[0], c.learning_rate[1], c.num_epochs)
    params = lasagne.layers.get_all_params(net['output'], trainable=True)
    updates = lasagne.updates.adam(train_loss, params, learning_rate=lr)
    val_acc = lasagne.objectives.binary_accuracy(output_det, label_var).mean()
    train_fn = theano.function([input_var, label_var], train_loss, updates=updates)
    val_fn = theano.function([input_var, label_var], [val_loss, val_acc])

    # init vars
    mve = None
    train_error={}
    val_error={}
    val_accuracy={}
    init_epoch = 0
    init_batch = 0
    num_train_batches = len(X_train) // c.batch_size
    num_val_batches = len(X_val) // c.batch_size


    # resume from epoch, batch
    print("Starting training...")
    [init_epoch, init_batch, mve, train_error, val_error, val_accuracy] = misc.resume(version, net['output'], fold=fold, seed=seed)
    if c.pretrain!=0 and init_epoch == 0 and init_batch == 0:
        print("load params from network: "+str(c.pretrain))
        prenet =  deepcopy(net)
        misc.load_last_params(prenet['output'], c.pretrain, best=True)
        model.match_net_params(anet, net)
    print("init epoch: "+str(init_epoch+1) + " batch: "+str(init_batch) +" best result: " +str(mve))

    for epoch in range(init_epoch, c.num_epochs):
        if epoch > init_epoch:
            init_batch = 0

        lr.set_value(lasagne.utils.floatX(lr_epoch[epoch]))
        train_err = 0
        train_batches = 0
        start_time = time.clock()
        val_err = 0
        val_acc = 0
        val_batches = 0
        
        # Training
        premod = 0
        for batch in getbatch(X_train, y_train, shape, c.batch_size, c.aug_params, shuffle=True, seed=epoch, init_batch=init_batch, augment=augment, resize=resize):
            inputs, targets = batch     
            train_err += train_fn(inputs, targets)
            # Save and print info every 10% of batches
            train_batches += 1
            mod = math.floor(train_batches / num_train_batches * 10) 
            if mod > premod and mod > 0:
                print(str(mod*10)+'%..',end="",flush=True)
                misc.save_progress(net['output'], version, epoch, train_batches, fold=fold, seed=seed)
            premod = mod
        # Training error
        if train_batches:
            train_error[epoch] = train_err / train_batches

        # Validation
        premod = 0
        for batch in getbatch(X_val, y_val, shape, c.batch_size, c.aug_params, shuffle=False, resize=resize):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            # Print info every 10% of batches
            val_batches += 1
            mod = math.floor(val_batches / num_val_batches * 10) 
            if mod > premod and mod > 0:
                print(str(mod*10)+'%..',end="",flush=True)
            premod = mod
        # Validation error
        val_error[epoch] = val_err / val_batches
        val_accuracy[epoch] = val_acc / val_batches * 100

        # Save parameters if epoch was best so far
        if mve is None or val_error[epoch] < mve:
            mve = val_error[epoch]
            misc.save_params(net['output'], version, epoch, best=True, fold=fold, seed=seed)

        # Print the results for this epoch
        misc.save_results(version, mve, train_error, val_error, val_accuracy, fold=fold, seed=seed)
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, c.num_epochs, time.clock() - start_time),flush=True)
        print("  training loss:\t\t{:.6f}".format(train_error[epoch]),flush=True)
        print("  validation loss:\t\t{:.6f}\t{:.6f}\t\t\t\t{:.6f}".format(val_error[epoch], mve, val_error[epoch]/train_error[epoch] ), flush=True)
        print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy[epoch]),flush=True)

        misc.finish(version, mve, fold, seed)




def main(): 
    try:
        print_GPU_memory()
    except:
        print("CPU used")

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", dest="version",  help="version", default=0.0)
    parser.add_argument("-cv", dest="cv",  help="cv", default=0) 
    parser.add_argument("-cvinv", dest="cvinv",  help="cvinv", default=0) 
    parser.add_argument("-fold", dest="fold",  help="fold", default=0) 
    parser.add_argument("-seed", dest="seed",  help="seed", default=1234)
    parser.add_argument("-train", dest="train",  help="train", default="train")
    options = parser.parse_args()

    version = int(options.version)
    seed = int(options.seed)
    cv = int(options.cv)
    cvinv = int(options.cvinv)
    fold = int(options.fold)
    train_dir = options.train


    print("Arguments:")
    print("----------")
    print(options)
    print()

    ifold = 0
    lfold = 1
    num_folds = 10
    step = 1

    if cv > 0:
        ifold = 1
        lfold = cv+1
        num_folds = lfold

    if cvinv > 0:
        ifold = cvinv
        lfold = 0
        step=-1
        num_folds = ifold

    if fold > 0:
        ifold = fold

    for fold in range(ifold, lfold, step):
        print("cv fold "+str(fold)+"\n")
        train_model(version = version, train_dir=train_dir, fold = fold, num_folds = num_folds, seed=seed)


if __name__ == '__main__':
    main()





