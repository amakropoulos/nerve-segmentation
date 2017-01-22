import lasagne 
from lasagne.layers import batch_norm as bn, MaxPool2DLayer, Upscale2DLayer, InputLayer
from lasagne.layers import ConcatLayer, DropoutLayer, Pool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import DenseLayer, NonlinearityLayer, GaussianNoiseLayer
from lasagne.layers import Conv2DLayer, InverseLayer, Deconv2DLayer, ReshapeLayer, TransformerLayer, FeaturePoolLayer, BiasLayer, PadLayer
from lasagne.nonlinearities import leaky_rectify
from lasagne.layers.base import MergeLayer
import numpy as np
import math


def print_network(network):
    print("Model:")
    print("------")
    layers = lasagne.layers.get_all_layers(network)
    for l in layers:
        if isinstance(l, lasagne.layers.normalization.BatchNormLayer) or isinstance(l, lasagne.layers.special.NonlinearityLayer) : continue
        print( ( str(type(l)).split("'")[1] ).split(".")[3]+"\t"+str(l.output_shape))
    print("")


def network(input_var, shape, version=0, filter_size=3, num_filters=8, depth=6, print_net=True, autoencoder=False, autoencoder_dropout=0.3):
    net = {}
    net['input'] = network = InputLayer(shape, input_var)

    if autoencoder: 
        network = lasagne.layers.dropout(network, autoencoder_dropout)

    import sys;
    sys.setrecursionlimit(40000)

    # default nonlinearities - change within model if needed
    nonlinearity=lasagne.nonlinearities.sigmoid
    convnonlinearity=lasagne.nonlinearities.rectify
        
    if version == 1: # segnet
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 

        for d in range(depth-1):
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
        
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 

    elif version == 2: # FCN
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)

        scale=1
        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=int(shape[1] * shape[2] * shape[3] / scale / scale ))
        net['fc%d'%len(net)] = network = DenseLayer(network, num_units=int(shape[1] * shape[2] * shape[3] / scale / scale ))
        net['rs%d'%len(net)] = network = ReshapeLayer(network, (-1, shape[1], int(shape[2]/ scale), int(shape[3] / scale)))
        net['output'] = network = NonlinearityLayer(network, nonlinearity)

    elif version == 3: # U-net 
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
            net['concat%d'%len(net)] = network = ConcatLayer( (network, net['conv%d'%convn]), axis=1, cropping=[None, None, 'center', 'center'])
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='same'); network = bn(network)
    
    net, network = fill_network(net, network, num_filters, filter_size, nonlinearity)
    if print_net: print_network(network)
    return net



def fill_network(net, network, num_filters=1, filter_size=3, nonlinearity=lasagne.nonlinearities.sigmoid, convnonlinearity=lasagne.nonlinearities.rectify):
    """perform extra convolutions with padd so the output is of the same size as input"""
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




######################## OTHER STUFF NOT CURRENTLY NEEDED ########################

def leaky_relu(x):
    import theano.tensor as T 
    return T.nnet.relu(x, alpha=1/3)



def match_net_params(anet, net):    
    """ Retrieve params of anet to and tranfer to net """
    for name in anet:
        if name not in net: continue
        l = anet[name]
        if isinstance(l, lasagne.layers.Conv2DLayer) or isinstance(l, lasagne.layers.MaxPool2DLayer):
            netl = net[name]
            lasagne.layers.set_all_param_values(netl, lasagne.layers.get_all_param_values(l))