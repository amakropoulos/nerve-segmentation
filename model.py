import lasagne 
from lasagne.layers import batch_norm as bn, MaxPool2DLayer, Upscale2DLayer, InputLayer
from lasagne.layers import ConcatLayer, DropoutLayer, Pool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import DenseLayer, NonlinearityLayer, GaussianNoiseLayer
from lasagne.layers import Conv2DLayer, InverseLayer, Deconv2DLayer, ReshapeLayer, TransformerLayer, FeaturePoolLayer, BiasLayer, PadLayer
from lasagne.nonlinearities import leaky_rectify
# from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.base import MergeLayer
import numpy as np
import math


def version_parameters(version):
    subversion = round(version%1.0*10)
    datadir = 'train'
    drop = False
    augment = 0.0
    dtbased = False
    shift = False
    nonlinearity=lasagne.nonlinearities.sigmoid

    modelversion = version

    if subversion == 1:
        datadir='train-reduced'
    elif subversion == 2:
        drop=True
    elif subversion == 3:
        augment=True
    elif subversion == 4:
        augment=True
        drop=True
    elif subversion == 5:
        augment=True
        dtbased=True
    
    if dtbased:
        nonlinearity = None

    modelversion = round(modelversion//1)

    print("Parameters:")
    print("-----------")
    print("version: "+str(version))
    print("modelversion: "+str(modelversion))
    print("datadir: "+str(datadir))
    print("augment: "+str(augment))
    print("shift: "+str(shift))
    print("dtbased: "+str(dtbased))
    print("drop: "+str(drop))
    print()

    return modelversion, dtbased, datadir, drop, augment, shift, nonlinearity



def print_network(network):
    print("Model:")
    print("------")
    layers = lasagne.layers.get_all_layers(network)
    for l in layers:
        if isinstance(l, lasagne.layers.normalization.BatchNormLayer) or isinstance(l, lasagne.layers.special.NonlinearityLayer) or isinstance(l, SumLayer): continue
        if isinstance(l, ZeroOutLayer):
            print("ZeroOutLayer"+"\t"+str(l.output_shape))
            continue;
        print( ( str(type(l)).split("'")[1] ).split(".")[3]+"\t"+str(l.output_shape))
    print("")


import theano.tensor as T 
def leaky_relu(x):
    return T.nnet.relu(x, alpha=1/3)



def match_net_params(anet, net):    
    for name in anet: 
        if name not in net: continue
        l = anet[name]
        if isinstance(l, lasagne.layers.Conv2DLayer) or isinstance(l, lasagne.layers.MaxPool2DLayer):
            netl = net[name]
            lasagne.layers.set_all_param_values(netl, lasagne.layers.get_all_param_values(l))




def autoencoder_network(input_var, shape, version=0, drop=False, nonlinearity=None, filter_size=3, num_filters=8, depth=6):
    net = {}
    net['input'] = network = InputLayer(shape, input_var)
    if drop: network = lasagne.layers.dropout(network, 0.3)

    if version == 0:
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)

        convn = len(net)-1
  
        for d in range(depth-1):
            net['dec%d'%len(net)] = network = InverseLayer(network, net['conv%d'%convn]); convn-=1; network = bn(network)
            net['ups%d'%len(net)] = network = InverseLayer(network, net['pool%d'%convn]); convn-=1   

        net['dec%d'%len(net)] = network = InverseLayer(network, net['conv%d'%convn]);
        net['output'] = network = NonlinearityLayer(network, nonlinearity)

    print_network(network)
    return net





class SumLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]


class ZeroOutLayer(MergeLayer):
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        inputs[1] = (inputs[1] >= 0.5)
        return inputs[0] * inputs[1][:,:,None,None]





def zero_out_network(input_var, shape, version=0, drop=False, nonlinearity=lasagne.nonlinearities.sigmoid, filter_size=3, num_filters=8, depth=6, print_net=True):
    net = {}
    net['input'] = network = InputLayer(shape, input_var)

    convnonlinearity = lasagne.nonlinearities.rectify

    if version == 0: # FCN
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 

        conv_shape = network.output_shape
        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=int(conv_shape[1] * conv_shape[2] * conv_shape[3]))
        # net['output'] = network = lasagne.layers.FlattenLayer( DenseLayer(network, num_units=1, nonlinearity=nonlinearity), 1 )
        net['output0'] = network = DenseLayer(network, num_units=1, nonlinearity=nonlinearity)
        net['output'] = network = ReshapeLayer(net['output0'], ([0], 1, 1, 1))

    if print_net: print_network(network)
    return net



def network(input_var, shape, version=0, drop=False, nonlinearity=lasagne.nonlinearities.sigmoid, filter_size=3, num_filters=8, depth=6, print_net=True):
    net = {}
    net['input'] = network = InputLayer(shape, input_var)

    import sys;
    sys.setrecursionlimit(40000)

    convnonlinearity = lasagne.nonlinearities.rectify
        
    if version == 32: # zero out network
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 

        while num_filters > 1:
            num_filters = max(int(num_filters/2), 1)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network=bn(network)

        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        net['output'] = network = ZeroOutLayer((net['input'], network))

    elif version == 31: # zero out network
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 

        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        net['output'] = network = ZeroOutLayer((net['input'], network))

    elif version == 30: # zero out network
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 

        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        net['output'] = network = ZeroOutLayer((net['input'], network))

    elif version == 21: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn -3
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['upconv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)             
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)

        while network.output_shape[-1] < shape[-1] - filter_size:
            num_filters = max(int(num_filters/2), 1)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network)
        net['output0'] = network = Conv2DLayer(network, num_filters=1, filter_size=filter_size, pad='full', nonlinearity=nonlinearity)

        net['output1'] = NonlinearityLayer( BiasLayer( SumLayer( ReshapeLayer(net['output0'], ([0], 1, -1)) ) ), nonlinearity=lasagne.nonlinearities.sigmoid)

        net['output'] = network = ZeroOutLayer((net['output0'], net['output1']))

    elif version == 20: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        convn = len(net)-1     

        net['zero-learn'] = ReshapeLayer( DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid) , ([0], 1))
        net['zero-out'] = network = ZeroOutLayer((network, net['zero-learn']))

        for d in range(depth-1):
            convn = convn -3
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['upconv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)             
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
        # net['output'] = net['output0']

    elif version == 18: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)
            if drop: network = lasagne.layers.dropout(network, 0.2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn-6
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)
            if drop: network = lasagne.layers.dropout(network, 0.2)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)


    elif version == 17: # U-net 
        convnonlinearity = leaky_rectify
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            # if drop: network = lasagne.layers.dropout(network, 0.2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn-4
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            if drop: network = lasagne.layers.dropout(network, 0.2)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same',nonlinearity=convnonlinearity); network = bn(network)
    

    elif version == 16: # U-net 
        pair = 3
        for d in range(depth-1):
            for p in range(pair):
                net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)

        for p in range(pair):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn-pair-1
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['upconv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            for p in range(pair):
                net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)

    elif version == 15: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn -3
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['upconv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    
    elif version == 14: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            
        for i in range(4):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
        
        convn = len(net)-3   
        for d in range(depth-1):
            convn = convn -3
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['upconv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1, pad='same'); network = bn(network)             
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    
    elif version == 13: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=3); num_filters=int(num_filters*2)
            if drop: network = lasagne.layers.dropout(network, 0.2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn-4
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=3); num_filters=int(num_filters/2)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            if drop: network = lasagne.layers.dropout(network, 0.2)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    

    elif version == 12: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            if drop: network = lasagne.layers.dropout(network, 0.2)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn-4
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            if drop: network = lasagne.layers.dropout(network, 0.2)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    
    elif version == 11: # U-net with 1x1 conv in the middle
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            if drop: network = lasagne.layers.dropout(network, 0.1)
        
        convn = len(net)-2   
        for i in range(4):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)

        for d in range(depth-1):
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            if drop: network = lasagne.layers.dropout(network, 0.1)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            convn = convn -3

    elif version == 6: # FCN
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)

        scale=1
        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=int(shape[1] * shape[2] * shape[3] / scale / scale ))
        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=int(shape[1] * shape[2] * shape[3] / scale / scale ))
        net['rs%d'%len(net)] = network = ReshapeLayer(network, (-1, shape[1], int(shape[2]/ scale), int(shape[3] / scale)))

        # scale = 4
        # net['fc%d'%len(net)] = network = DenseLayer(network, num_units=int(shape[1] * shape[2] * shape[3] / scale / scale ))
        # net['rs%d'%len(net)] = network = ReshapeLayer(network, (-1, shape[1], int(shape[2]/ scale), int(shape[3] / scale)))
        # net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=scale)
        net['output'] = network = NonlinearityLayer(network, nonlinearity)

    elif version == 3: # U-net 
        for d in range(depth-1):
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            if drop: network = lasagne.layers.dropout(network, 0.1)
            
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
        convn = len(net)-1     
        for d in range(depth-1):
            convn = convn -3
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            if drop: network = lasagne.layers.dropout(network, 0.1)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    
    elif version == 0: # segnet
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            if drop: network = lasagne.layers.dropout(network, 0.2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
  
        if drop: network = lasagne.layers.dropout(network, 0.5)

        for d in range(depth-1):
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            if drop: network = lasagne.layers.dropout(network, 0.2)
        
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 
    net, network = fill_network(net, network, num_filters, filter_size, drop, nonlinearity, convnonlinearity)
    if print_net: print_network(network)
    return net


def fill_network(net, network, num_filters=1, filter_size=3, drop=False, nonlinearity=lasagne.nonlinearities.sigmoid, convnonlinearity=lasagne.nonlinearities.rectify):
    img_shape = None
    layers = lasagne.layers.get_all_layers(network)
    for l in layers:
        if isinstance(l, lasagne.layers.InputLayer):
            img_shape = l.output_shape

    if network.output_shape[-1] == img_shape[-1]:
        return net, network

    while network.output_shape[-1] < img_shape[-1]-filter_size:
        num_filters = max(int(num_filters/2), 1)
        if num_filters == 1:
            break
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full',nonlinearity=convnonlinearity); network=bn(network)

    net['output0'] = network = Conv2DLayer(network, num_filters=1, filter_size=filter_size, pad='full', nonlinearity=nonlinearity)

    if network.output_shape[-2] < img_shape[-2] or network.output_shape[-1] < img_shape[-1]:
        width = [int( (img_shape[-2]-network.output_shape[-2])/2 ), int( (img_shape[-1]-network.output_shape[-1])/2 )]
        net['pad'] = network = PadLayer(network, width)

    net['output'] = network
    return net, network






# def fill_network(net, network, num_filters=1, filter_size=3, drop=False, nonlinearity=lasagne.nonlinearities.sigmoid, convnonlinearity=lasagne.nonlinearities.rectify):
#     img_shape = None
#     layers = lasagne.layers.get_all_layers(network)
#     for l in layers:
#         if isinstance(l, lasagne.layers.InputLayer):
#             img_shape = l.output_shape

#     if network.output_shape[-1] == img_shape[-1]:
#         return net, network

#     while network.output_shape[-1] < img_shape[-1]-filter_size:
#         num_filters = max(int(num_filters/2), 1)
#         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full',nonlinearity=convnonlinearity); network=bn(network)

#     net['output'] = network = Conv2DLayer(network, num_filters=1, filter_size=filter_size, pad='full', nonlinearity=nonlinearity)
#     return net, network







    # if version == 4:
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 
        
    #     for d in range(depth-1):
    #         net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)
  
    #     for d in range(depth-1):
    #         net['dec%d'%len(net)] = network = bn(Deconv2DLayer(network, num_filters=num_filters, filter_size=filter_size))
    #         net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            
    #     net['conv%d'%len(net)] = network = bn(Deconv2DLayer(network, num_filters=num_filters, filter_size=filter_size))
    #     net['conv%d'%len(net)] = network = bn(Deconv2DLayer(network, num_filters=1, filter_size=filter_size))
    #     net['conv%d'%len(net)] = network = bn(Deconv2DLayer(network, num_filters=1, filter_size=filter_size))
    #     net['output'] = network = bn(Deconv2DLayer(network, num_filters=1, filter_size=filter_size))


    # elif version == 3:
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); 
    #     net['bn%d'%len(net)] = network = bn(network) 
        
    #     for d in range(depth-1):
    #         net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); 
    #         net['bn%d'%len(net)] = network = bn(network)

    #     convn = len(net)-2
  
    #     for d in range(depth-1):
    #         net['dec%d'%len(net)] = network = InverseLayer(network, net['conv%d'%convn]); convn-=1
    #         net['bn%d'%len(net)] = network = bn(network)
    #         net['ups%d'%len(net)] = network = InverseLayer(network, net['pool%d'%convn]); convn-=2   

    #     net['dec%d'%len(net)] = network = InverseLayer(network, net['conv%d'%convn]); convn-=1
    #     net['output'] = network = NonlinearityLayer(network, nonlinearity)



    # if version == 14: # U-net 
    #     for d in range(depth-1):
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
    #         net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
    #         if drop: network = lasagne.layers.dropout(network, 0.1)
            
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
        
    #     convn = len(net)-1     
    #     for d in range(depth-1):
    #         convn = convn -4
    #         net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
    #         net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
    #         if drop: network = lasagne.layers.dropout(network, 0.1)
    #         net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         if d < depth-2:
    #             net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
    
    # elif version == 13: # U-net 
    #     for d in range(depth-1):
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
    #         if drop: network = lasagne.layers.dropout(network, 0.1)
            
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
    #     net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network) 
        
    #     convn = len(net)-1     
    #     for d in range(depth-1):
    #         convn = convn -4
    #         net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
    #         net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
    #         if drop: network = lasagne.layers.dropout(network, 0.1)
    #         net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=1); network = bn(network)
    #         net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    #         net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)