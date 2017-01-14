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


def dice(pred, tgt, ss=1):
    return -2*(T.sum(pred*tgt)+ss)/(T.sum(pred) + T.sum(tgt) + ss)

def dice_real(pred, tgt):
    return dice(pred, tgt)

def dice_predict(pred, tgt):
    predeq = (pred >= 0.5)
    tgteq = (tgt >= 0.5)
    den = predeq.sum() + tgteq.sum()
    if den == 0: return -1
    return -2* (predeq*tgteq).sum()/den


def getbatch(inputs, targets, batchsize, shuffle=False, seed=-1, init_batch=0, augment=False, resize=0):
    indices = np.arange(len(inputs))
    if shuffle:
        if seed>=0: 
            np.random.seed(seed)
        np.random.shuffle(indices)

    shape = (None, 1, c.height, c.width) 
    if resize:
        shape = (None, 1, round(c.height*resize), round(c.width*resize) ) 

    Xs = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    ys = np.zeros((batchsize, 1, shape[2], shape[3]), dtype='float32')
    d=0
    for ii in range(init_batch*batchsize, len(inputs)):
        i = ii
        if shuffle:
            i = indices[ii]
        img = misc.load_image(inputs[i])
        lbl = misc.load_image(targets[i])
        if augment:
            img, lbl = misc.augment_image(img, lbl)
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


def print_GPU_memory():
    import theano.sandbox.cuda.basic_ops as sbcuda
    GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
    print("GPUFreeMemory: "+str(GPUFreeMemoryInBytes))
    print("")


# def plot_errors(version=0):
#     mve, train_error, val_error, val_accuracy = misc.load_results(version)
#     train_loss = []
#     for value in train_error.values(): train_loss.append(abs(float(value)))
#     valid_loss = []
#     for value in val_error.values(): valid_loss.append(abs(float(value)))
#     valid_ratio = np.divide(valid_loss,train_loss)
    
#     pyplot.subplots(1, 2, sharey=True)
#     ax = pyplot.subplot(2, 1, 1)
#     ax.set_ylim((0, 1))
#     pyplot.plot(train_loss, linewidth=3, label="train")
#     pyplot.plot(valid_loss, linewidth=3, label="valid")
#     pyplot.grid()
#     pyplot.legend(loc=4)
#     pyplot.xlabel("epoch")
#     pyplot.ylabel("loss")

#     pyplot.subplot(2, 1, 2)
#     pyplot.plot(valid_ratio, linewidth=3, label="valid")
#     pyplot.show()


def print_errors(version=0, cv=0, seed=1234):
    [weight, dum1, dum2, dum3] = misc.load_results(version, cv=cv, seed=seed)
    print("score: "+str(weight))



def add_regularization(network):
    import lasagne.regularization as reg
    layers = lasagne.layers.get_all_layers(network)
    penalty_layers = []
    for l in layers:
        if isinstance(l, lasagne.layers.Conv2DLayer):
            penalty_layers.append(l)
    return reg.regularize_layer_params(penalty_layers, reg.l2)


def show_model(version=0, depth=6, filters=8, filter_size=3, showdir='show', resize=0, validation=0.1, cv=0, thresholds=False, minsize=5000, seed=1234, save=False):
    modelversion, dtbased, datadir, drop, augment, shift, nonlinearity = model.version_parameters(version)
    datadir = 'train-orig'

    orig_shape = (1, 1, c.height, c.width)
    if resize != 0:
        shape = (1, 1, round(c.height*resize), round(c.width*resize) ) 
    else:
        shape = (1, 1, c.height, c.width)

    input_var = T.tensor4('input')
    label_var = T.tensor4('label')

    net = model.network(input_var, shape, filter_size=filter_size, version=modelversion, drop=drop, nonlinearity=nonlinearity, depth=depth, num_filters=filters, print_net=False)
  
    print("Loading data..")
    start_time = time.clock()
    X_train, y_train, X_val, y_val = misc.load_data(val_pct=validation, datadir=datadir, cv=cv)
    print("took " + str(time.clock()-start_time )+"s")


    output_det = lasagne.layers.get_output(net['output'], deterministic=True)
    predict_model = theano.function(inputs=[input_var], outputs=output_det)

    misc.load_last_params(net['output'], version, best=True, cv=cv, seed=seed)

    print_idx = 0
    print_dir = os.path.join(showdir, 'v{}'.format(version))
    if cv > 0:
        print_dir = os.path.join(print_dir, 'cv{}'.format(cv))
    else:
        print_dir = os.path.join(print_dir, 'seed{}'.format(seed))
    if not os.path.exists(print_dir):
        os.makedirs(print_dir)

    if not thresholds:
        val_err = 0
    else:
        val_err = {}
        thrs = np.linspace(0,6500,66)
        for t in thrs: val_err[t] = 0

    val_batches = 0
    num_batches = len(X_val)
    premod = 0


    for batch in getbatch(X_val, y_val, 1):
        img, tgt = batch
        if resize:
            img = cv2.resize(img[0][0], (shape[3], shape[2]), interpolation=cv2.INTER_CUBIC).reshape(shape)
        pred = predict_model(img)
        if resize:
            pred = cv2.resize(pred[0][0], (orig_shape[3], orig_shape[2]), interpolation=cv2.INTER_LINEAR).reshape(orig_shape)

        if not thresholds:
            pred = test.postprocess(pred, minsize)
            val_err += dice_predict(pred, tgt)
        else:
            for t in thrs: 
                predt = test.postprocess(pred, t)
                val_err[t] += dice_predict(predt, tgt)

        if not thresholds and save:
            base = os.path.basename(X_val[val_batches]).split('.')[0]
            fnpred = os.path.join(print_dir, '{}.tif'.format(base))
            scipy.misc.imsave(fnpred, pred.reshape(c.height, c.width) * np.float32(255))

        val_batches += 1
        
        mod = math.floor(val_batches / num_batches * 10) 
        if mod > premod and mod > 0:
            print(str(mod*10)+'%..',end="",flush=True)
        premod = mod

    if thresholds:
        for t in thrs: 
            val_err[t] = val_err[t] / val_batches
            print(str(t)+": "+str(val_err[t]))
    else:
        val_error = val_err / val_batches
        print("  validation loss:\t\t{:.6f}".format(val_error))
    print("Prediction took {:.3f}s".format(time.clock() - start_time))




def train_model(version=0, depth=6, filters=8, filter_size=3, regularization=0, test=False, resize=0, validation=0.1, cv=0, learning_rate=[3e-3], seed=1234):
    modelversion, dtbased, datadir, drop, augment, shift, nonlinearity = model.version_parameters(version)

    batch_size = c.batch_size
    if resize != 0:
        shape = (None, 1, round(c.height*resize), round(c.width*resize) ) 
    else:
        shape = (None, 1, c.height, c.width)

    print("Loading data..")
    start_time = time.clock()   
    if(test):
        X_train, y_train, X_val, y_val = misc.load_data(val_pct=validation, first=3, datadir=datadir, seed=seed)
    else:
        X_train, y_train, X_val, y_val = misc.load_data(val_pct=validation, datadir=datadir, cv=cv, seed=seed)
    print("took " + str(time.clock()-start_time )+"s")
    num_train_batches = len(X_train) // batch_size
    num_val_batches = len(X_val) // batch_size

    [epoch, batch] = misc.completed(version, cv, seed=seed)
    if epoch==c.num_epochs-1 and batch>=num_train_batches:
        print_errors(version, cv)
        return


    input_var = T.tensor4('input')
    label_var = T.tensor4('label')
    net = model.network(input_var, shape, filter_size=filter_size, version=modelversion, drop=drop, nonlinearity=nonlinearity, depth=depth, num_filters=filters)
    

    output = lasagne.layers.get_output(net['output'])
    output_det = lasagne.layers.get_output(net['output'], deterministic=True)
    loss = dice(output, label_var)
    train_loss = dice_real(output, label_var)
    val_loss = dice_real(output_det, label_var)

    if regularization>0:
         loss = loss + regularization*add_regularization(net['output'])

    lr = theano.shared(lasagne.utils.floatX(learning_rate[0]))
    lr_epoch = np.linspace(learning_rate[0], learning_rate[1], c.num_epochs)
    params = lasagne.layers.get_all_params(net['output'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    val_acc = lasagne.objectives.binary_accuracy(output_det, label_var).mean()
    train_fn = theano.function([input_var, label_var], train_loss, updates=updates)
    val_fn = theano.function([input_var, label_var], [val_loss, val_acc])


    mve = None
    train_error={}
    val_error={}
    val_accuracy={}
    init_epoch = 0
    init_batch = 0

    if(test):
        train_batches = 0
        exe_time = 0
        for batch in getbatch(X_train, y_train, batch_size, shuffle=True, seed=0, init_batch=init_batch, augment=augment, resize=resize):
            start = time.clock()
            inputs, targets = batch     
            loss = train_fn(inputs, targets)
            train_batches += 1
            exe_time += time.clock()-start
            print(str(loss)+" "+str(exe_time/train_batches))
            if train_batches == 10: break
        return

    [init_epoch, init_batch, mve, train_error, val_error, val_accuracy] = misc.resume(net['output'], version, cv=cv, seed=seed)
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
        
        for batch in getbatch(X_train, y_train, batch_size, shuffle=True, seed=epoch, init_batch=init_batch, augment=augment, resize=resize):
            inputs, targets = batch     
            train_err += train_fn(inputs, targets)
            train_batches += 1
            mod = math.floor(train_batches / num_train_batches * 10) 
            if mod > premod and mod > 0:
                print(str(mod*10)+'%..',end="",flush=True)
                misc.save_progress(net['output'], version, epoch, train_batches, cv=cv, seed=seed)
            premod = mod
        if train_batches:
            train_error[epoch] = train_err / train_batches

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        premod = 0
        for batch in getbatch(X_val, y_val, c.batch_size, shuffle=False, resize=resize):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            mod = math.floor(val_batches / num_val_batches * 10) 
            if mod > premod and mod > 0:
                print(str(mod*10)+'%..',end="",flush=True)
            premod = mod
        val_error[epoch] = val_err / val_batches
        val_accuracy[epoch] = val_acc / val_batches * 100

        # Then we print the results for this epoch:
        if mve is None or val_error[epoch] < mve:
            mve = val_error[epoch]
            misc.save_params(net['output'], version, epoch, best=True, cv=cv, seed=seed)

        misc.save_results(version, mve, train_error, val_error, val_accuracy, cv=cv, seed=seed)
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
    parser.add_argument("-seed", dest="seed",  help="seed", default=1234)

    parser.add_argument("--test", dest="test",  help="test", action='store_true')
    parser.add_argument("--plot", dest="plot",  help="plot", action='store_true')
    parser.add_argument("--show", dest="show",  help="show", action='store_true')
    parser.add_argument("--show-final", dest="show_final",  help="show_final", action='store_true')
    parser.add_argument("--show-save", dest="show_save",  help="show_save", action='store_true')
    parser.add_argument("--print", dest="print_err",  help="print_err", action='store_true')
    options = parser.parse_args()

    version = float(options.version)
    depth = int(options.depth)
    filters = int(options.filters)
    test = int(options.test)
    plot = int(options.plot)
    show = int(options.show)
    show_final = int(options.show_final)
    show_save = int(options.show_save)
    print_err = int(options.print_err)
    seed = int(options.seed)

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
            if show:
                show_model(version, depth, filters, filter_size, resize=resize, validation=validation, cv=fold, thresholds=True, seed=seed)
            elif show_final:
                show_model(version, depth, filters, filter_size, resize=resize, validation=validation, cv=fold, thresholds=False, minsize=minsize, seed=seed, save=show_save)
            elif print_err:
                print_errors(version, fold, seed=seed)              
            else:
                train_model(version, depth, filters, filter_size, regularization, test=test, resize=resize, validation=validation, learning_rate=learning_rate, cv=fold, seed=seed)


if __name__ == '__main__':
    main()





