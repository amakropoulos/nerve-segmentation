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




def getbatch(inputs, predictions, targets, batchsize, shuffle=False, seed=-1, init_batch=0, augment=False, resize=0):
    indices = np.arange(len(inputs))
    if shuffle:
        if seed>=0: 
            np.random.seed(seed)
        np.random.shuffle(indices)

    shape = (None, 1, c.height, c.width) 
    if resize:
        shape = (None, 1, round(c.height*resize), round(c.width*resize) ) 

    Xs = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    Ps = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    ys = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    d=0
    for ii in range(init_batch*batchsize, len(inputs)):
        i = ii
        if shuffle:
            i = indices[ii]
        img = misc.load_image(inputs[i])
        pred = misc.load_image(predictions[i])
        lbl = misc.load_image(targets[i])
        # if augment:
        #     img, pred, lbl = misc.augment_image3(img, pred, lbl)
        if resize:
            img = cv2.resize(img[0], (shape[3], shape[2]), interpolation=cv2.INTER_CUBIC).reshape(shape[1:])
            pred = cv2.resize(pred[0], (shape[3], shape[2]), interpolation=cv2.INTER_NEAREST).reshape(shape[1:])
            lbl = cv2.resize(lbl[0], (shape[3], shape[2]), interpolation=cv2.INTER_NEAREST).reshape(shape[1:])
        Xs[d] = img
        Ps[d] = pred
        ys[d] = lbl
        d+=1
        if d == batchsize:
            d = 0
            yield Xs, Ps, ys
    if(d>0):
        yield Xs[0:d,:,:,:], Ps[0:d,:,:,:], ys[0:d,:,:,:]


def train_model(version=0, depth=6, filters=8, filter_size=3, regularization=0, test=False, resize=0, validation=0.1, cv=0, learning_rate=[3e-3]):
    modelversion, dtbased, datadir, drop, augment, shift, nonlinearity = model.version_parameters(version)

    batch_size = c.batch_size
    if resize != 0:
        shape = (None, 1, round(c.height*resize), round(c.width*resize) ) 
    else:
        shape = (None, 1, c.height, c.width)

    input_var = T.tensor4('input')
    pred_var  = T.tensor4('pred')]
    label_var = T.tensor4('label')
    net = model.zero_out_network(input_var, pred_var, shape, filter_size=filter_size, version=modelversion, drop=drop, nonlinearity=nonlinearity, depth=depth, num_filters=filters)
    
    print("Loading data..")
    start_time = time.clock()   
    X_train, y_train, X_val, y_val = misc.load_data(val_pct=validation, datadir=datadir, cv=cv)
    print("took " + str(time.clock()-start_time )+"s")


    output = lasagne.layers.get_output(net['output'])
    output_det = lasagne.layers.get_output(net['output'], deterministic=True)
    loss = train.dice(output, label_var)
    train_loss = train.dice_real(output, label_var)
    val_loss = train.dice_real(output_det, label_var)

    if regularization>0:
         loss = loss + regularization*train.add_regularization(net['output'])

    lr = theano.shared(lasagne.utils.floatX(learning_rate[0]))
    lr_epoch = np.linspace(learning_rate[0], learning_rate[1], c.num_epochs)
    params = lasagne.layers.get_all_params(net['output'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    val_acc = lasagne.objectives.binary_accuracy(output_det, label_var).mean()
    train_fn = theano.function([input_var, pred_var, label_var], train_loss, updates=updates)
    val_fn = theano.function([input_var, pred_var, label_var], [val_loss, val_acc])


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
        for batch in getbatch(X_train, y_train, batch_size, shuffle=True, seed=epoch, init_batch=init_batch, augment=augment, resize=resize):
            inputs, predictions, targets = batch     
            train_err += train_fn(inputs, predictions, targets)
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
        for batch in getbatch(X_val, y_val, c.batch_size, shuffle=False, resize=resize):
            inputs, predictions, targets = batch
            err, acc = val_fn(inputs, predictions, targets)
            val_err += err
            val_acc += acc
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

    parser.add_argument("--test", dest="test",  help="test", action='store_true')
    parser.add_argument("--plot", dest="plot",  help="plot", action='store_true')
    parser.add_argument("--show", dest="show",  help="show", action='store_true')
    parser.add_argument("--show-final", dest="show_final",  help="show_final", action='store_true')
    options = parser.parse_args()

    version = float(options.version)
    depth = int(options.depth)
    filters = int(options.filters)
    test = int(options.test)
    plot = int(options.plot)
    show = int(options.show)
    show_final = int(options.show_final)
    regularization = float(options.regularization)
    cv = int(options.cv)
    cvinv = int(options.cvinv)
    cvi = int(options.cvi)
    resize = float(options.resize)
    validation = float(options.validation)
    filter_size = int(options.filter_size)
    learning_rate = [float(options.learning_rate), float(options.learning_rate2)]

    print("Arguments:")
    print("----------")
    print(options)
    print()

    if plot:
        plot_errors(version)
    else:
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
            train_model(version, depth, filters, filter_size, regularization, test=test, resize=resize, validation=validation, learning_rate=learning_rate, cv=fold)


if __name__ == '__main__':
    main()


