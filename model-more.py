
    elif version == 10: # seg-net model with 2 scale path-ways
        orig_num_filters = num_filters
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        b = b.flatten()  # identity transform
        W = lasagne.init.Constant(0.0)
        l_loc = lasagne.layers.DenseLayer(network, num_units=6, W=W, b=b, nonlinearity=None)

        # net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
        net['trans%d'%len(net)] = network = TransformerLayer(network, l_loc, downsample_factor=2)
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
        
        for d in range(depth-2):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network=bn(network) 
  
        if drop: network = lasagne.layers.dropout(network, 0.5)

        for d in range(depth-2):
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)

        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 
        net['trans%d'%len(net)] = network = TransformerLayer(network, l_loc, downsample_factor=0.5)

        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network=bn(network) 
        # net, network = fill_network(net, network, num_filters, filter_size, drop, nonlinearity, convnonlinearity)
        net['output1'] = network



        num_filters = orig_num_filters
        net['2conv%d'%len(net)] = network2 = Conv2DLayer(net['input'], num_filters=num_filters, filter_size=filter_size); network2=bn(network2) 
        
        for d in range(depth-1):
            net['2pool%d'%len(net)] = network2 = MaxPool2DLayer(network2, pool_size=2); num_filters=int(num_filters*2)
            net['2conv%d'%len(net)] = network2 = Conv2DLayer(network2, num_filters=num_filters, filter_size=filter_size); network2=bn(network2) 
  
        if drop: network2 = lasagne.layers.dropout(network2, 0.5)

        for d in range(depth-1):
            net['2dec%d'%len(net)] = network2 = Conv2DLayer(network2, num_filters=num_filters, filter_size=filter_size, pad='full'); network2=bn(network2) 
            net['2ups%d'%len(net)] = network2 = Upscale2DLayer(network2, scale_factor=2); num_filters=int(num_filters/2)
        
        net['2conv%d'%len(net)] = network2 = Conv2DLayer(network2, num_filters=num_filters, filter_size=filter_size, pad='full'); network2=bn(network2) 
        # net, network2 = fill_network(net, network2, num_filters, filter_size, drop, nonlinearity, convnonlinearity)
        net['output2'] = network2



        net['concat%d'%len(net)] = network = ConcatLayer( (net['output1'], net['output2']), axis=1, cropping=[None, None, 'center', 'center'])
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=orig_num_filters, filter_size=filter_size, pad='full'); network=bn(network) 


    elif version == 7: # use leaky rely
        convnonlinearity = leaky_relu
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, nonlinearity=convnonlinearity); network = bn(network)
        
        for d in range(depth-1):
            net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, nonlinearity=convnonlinearity); network = bn(network)
  
        for d in range(depth-1):
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full', nonlinearity=convnonlinearity); network = bn(network)
            net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
        
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full', nonlinearity=convnonlinearity); network = bn(network)

    elif version == 8: # segnet, add some more layers
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network) 
        
        for d in range(2, depth+1):
            if d<=6:
                net['pool%d'%len(net)] = network = MaxPool2DLayer(network, pool_size=2); num_filters=int(num_filters*2)
            if drop: network = lasagne.layers.dropout(network, 0.1)
            net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size); network = bn(network)
  
        if drop: network = lasagne.layers.dropout(network, 0.5)

        for d in range(depth,1,-1):
            net['dec%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network = bn(network)
            if d<=6:
                net['ups%d'%len(net)] = network = Upscale2DLayer(network, scale_factor=2); num_filters=int(num_filters/2)
            if drop: network = lasagne.layers.dropout(network, 0.1)
        
        net['conv%d'%len(net)] = network = Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, pad='full'); network = bn(network)
