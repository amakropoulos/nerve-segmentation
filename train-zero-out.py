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
from matplotlib import pyplot
import cv2
from skimage import measure
import glob

import config as c
import model
import misc
import test
import train


def dice(zero_pred, pred, tgt, ss=1):
    pred = T.mul(pred, zero_pred)
    return -2*(T.sum(pred*tgt)+ss)/(T.sum(pred) + T.sum(tgt) + ss)


def train_model(version=0, depth=6, filters=8, filter_size=3, regularization=0, resize=0, validation=0.1, cv=0, learning_rate=[3e-3]):
    modelversion, dtbased, datadir, drop, augment, shift, nonlinearity = model.version_parameters(version)

    batch_size = c.batch_size
    if resize != 0:
        shape = (None, 1, round(c.height*resize), round(c.width*resize) ) 
    else:
        shape = (None, 1, c.height, c.width)

    input_var = T.tensor4('input')
    target_var = T.tensor4('target')
    # label_var = T.vector('label')
    label_var = T.tensor4('label')
    net = model.zero_out_network(input_var, shape, filter_size=filter_size, version=modelversion, drop=drop, nonlinearity=nonlinearity, depth=depth, num_filters=filters)
    
    print("Loading data..")
    start_time = time.clock()   
    X_train, y_train, X_val, y_val = misc.load_data(val_pct=validation, datadir=datadir, cv=cv)
    print("took " + str(time.clock()-start_time )+"s")


    output = lasagne.layers.get_output(net['output'])
    output_det = lasagne.layers.get_output(net['output'], deterministic=True)
    # loss = lasagne.objectives.binary_crossentropy(output, label_var).mean()
    # train_loss = lasagne.objectives.binary_crossentropy(output, label_var).mean()
    # val_loss = lasagne.objectives.binary_crossentropy(output_det, label_var).mean()

    loss = dice(output, input_var, target_var)
    train_loss = dice(output, input_var, target_var)
    val_loss = dice(output_det, input_var, target_var)


    if regularization>0:
         loss = loss + regularization*add_regularization(net['output'])

    lr = theano.shared(lasagne.utils.floatX(learning_rate[0]))
    lr_epoch = np.linspace(learning_rate[0], learning_rate[1], c.num_epochs)
    params = lasagne.layers.get_all_params(net['output'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # val_acc = lasagne.objectives.binary_accuracy(output_det, label_var).mean()
    # train_fn = theano.function([input_var, label_var], train_loss, updates=updates)
    # val_fn = theano.function([input_var, label_var], [val_loss, val_acc])
    train_fn = theano.function([input_var, target_var], train_loss, updates=updates)
    val_fn = theano.function([input_var, target_var], val_loss)


    mve = None
    train_error={}
    val_error={}
    val_accuracy={}
    init_epoch = 0
    init_batch = 0
    [init_epoch, init_batch, mve, train_error, val_error, val_accuracy] = misc.resume(net['output'], version, cv=cv)
    print("init epoch: "+str(init_epoch+1) + " batch: "+str(init_batch) +" best result: " +str(mve))
        

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(init_epoch, c.num_epochs):
        if epoch > init_epoch:
            init_batch = 0

        lr.set_value(lasagne.utils.floatX(lr_epoch[epoch]))
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.clock()
        premod = 0
        num_batches = len(X_train) // batch_size
        for batch in train.getbatch(X_train, y_train, batch_size, shuffle=True, seed=epoch, init_batch=init_batch, augment=augment, resize=resize):
            inputs, targets = batch     
            zero_targets = np.reshape(targets, (targets.shape[0], -1))
            zero_targets = np.sum(zero_targets, axis=-1)
            zero_targets = lasagne.utils.floatX(zero_targets > 0)
            train_err += train_fn(inputs, targets)
            train_batches += 1
            mod = math.floor(train_batches / num_batches * 10) 
            if mod > premod and mod > 0:
                print(str(mod*10)+'%..',end="",flush=True)
                misc.save_progress(net['output'], version, epoch, train_batches, cv=cv)
            premod = mod
        if train_batches:
            train_error[epoch] = train_err / train_batches

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        num_batches = len(X_val) // batch_size
        premod = 0
        for batch in train.getbatch(X_val, y_val, c.batch_size, shuffle=False, resize=resize):
            inputs, targets = batch
            zero_targets = np.reshape(targets, (targets.shape[0], -1))
            zero_targets = np.sum(zero_targets, axis=-1)
            zero_targets = lasagne.utils.floatX(zero_targets > 0)
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
            mod = math.floor(val_batches / num_batches * 10) 
            if mod > premod and mod > 0:
                print(str(mod*10)+'%..',end="",flush=True)
            premod = mod
        val_error[epoch] = val_err / val_batches
        val_accuracy[epoch] = val_acc / val_batches * 100

        # Then we print the results for this epoch:
        if mve is None or val_error[epoch] < mve:
            mve = val_error[epoch]
            misc.save_params(net['output'], version, epoch, best=True, cv=cv)

        misc.save_results(version, mve, train_error, val_error, val_accuracy, cv=cv)
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, c.num_epochs, time.clock() - start_time),flush=True)
        print("  training loss:\t\t{:.6f}".format(train_error[epoch]),flush=True)
        print("  validation loss:\t\t{:.6f}\t{:.6f}\t\t\t\t{:.6f}".format(val_error[epoch], mve, val_error[epoch]/train_error[epoch] ), flush=True)
        print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy[epoch]),flush=True)



def main(): 
    try:
        print_GPU_memory()
    except:
        print("CPU used")

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", dest="version",  help="version", default=0.0)
    parser.add_argument("-d", "--depth", dest="depth",  help="depth", default=6)
    parser.add_argument("-f", "--filters", dest="filters",  help="filters", default=8)
    parser.add_argument("-fs", "--filters-size", dest="filter_size",  help="filter_size", default=3)
    parser.add_argument("-reg", "--regularization", dest="regularization",  help="regularization", default=0.0) #1e-3 - 1e-2
    parser.add_argument("-resize", dest="resize",  help="resize", default=0.0) 
    parser.add_argument("-validation", dest="validation",  help="validation", default=0.1) 
    parser.add_argument("-cv", dest="cv",  help="cv", default=0) 
    parser.add_argument("-cvinv", dest="cvinv",  help="cvinv", default=0) 
    parser.add_argument("-cvi", dest="cvi",  help="cvi", default=0) 
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate",  help="learning_rate", default=3e-3) 
    parser.add_argument("-lr2", "--learning-rate2", dest="learning_rate2",  help="learning_rate2", default=3e-3) 
    parser.add_argument("-minsize", dest="minsize",  help="minsize", default=5000) 
    options = parser.parse_args()

    version = float(options.version)
    depth = int(options.depth)
    filters = int(options.filters)
    regularization = float(options.regularization)
    cv = int(options.cv)
    cvinv = int(options.cvinv)
    cvi = int(options.cvi)
    resize = float(options.resize)
    validation = float(options.validation)
    filter_size = int(options.filter_size)
    learning_rate = [float(options.learning_rate), float(options.learning_rate2)]
    minsize = int(options.minsize)

    print("Arguments:")
    print("----------")
    print(options)
    print()

    ifold = 0
    nfold = 1
    step = 1

    if cv > 0:
        validation = 1/cv
        ifold = 1
        nfold = cv+1

    if cvi > 0:
        ifold = cvi
        nfold = cvi+1

    if cvinv > 0:
        validation = 1/cvinv
        ifold = cvinv
        nfold = 0
        step=-1

    print("validation "+str(validation))
    for fold in range(ifold, nfold, step):
        print("cv fold "+str(fold)+"\n")
        train_model(version, depth, filters, filter_size, regularization, resize=resize, validation=validation, learning_rate=learning_rate, cv=fold)


if __name__ == '__main__':
    main()

