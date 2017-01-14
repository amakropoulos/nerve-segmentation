

def show_model_tta(version=0, depth=6, filters=8, filter_size=3, showdir='show', resize=0, validation=0.1, cv=0, thresholds=False, minsize=5000, tta_num=10):
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

    misc.load_last_params(net['output'], version, best=True, cv=cv)

    print_idx = 0
    print_dir = os.path.join(showdir, 'v{}'.format(version))
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

        pred = tgt * 0
        for tt in range(tta_num):
            if tt == 0:
                tta_pred = predict_model(img)
            else:
                tta_pred = misc.tta(img,predict_model)
            if resize:
                tta_pred = cv2.resize(tta_pred[0][0], (orig_shape[3], orig_shape[2]), interpolation=cv2.INTER_LINEAR).reshape(orig_shape)
            # tta_pred = test.postprocess(tta_pred, minsize)
            pred += tta_pred
        pred/=tta_num

        # pred = test.postprocess(pred, 0)
        pred = test.postprocess(pred, minsize)
        val_err += dice_predict(pred, tgt)

        # if not thresholds:
        #     base = os.path.basename(X_val[val_batches]).split('.')[0]
        #     # fnpred = os.path.join(print_dir, '{}_pred.tif'.format(base))
        #     # fntgt = os.path.join(print_dir, '{}_tgt.tif'.format(base))
        #     fnpred = os.path.join(print_dir, '{}-tta.tif'.format(base))
        #     scipy.misc.imsave(fnpred, pred.reshape(c.height, c.width) * np.float32(255))
        #     # scipy.misc.imsave(fntgt, tgt.reshape(c.height, c.width) * np.float32(255))
            
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








    # # MRF
    # import pickle
    # alpha = 0.3
    # val_err = 0
    # val_batches = 0
    # dim = (orig_shape[3], orig_shape[2])
    # for batch in getbatch(X_val, y_val, 1, orig_shape, stride, dtbased=dtbased):
    #     img, tgt = batch
    #     base = os.path.basename(X_val[val_batches]).split('.')[0]
    #     close_warp_names = glob.glob("warps/pass/"+base+"-*.pickle")
    #     fnpred = os.path.join(print_dir, '{}_pred.tif'.format(base))
    #     pred = misc.load_image(fnpred)
    #     mrfsum = None
    #     mrfden = 0
    #     for warp in close_warp_names:
    #         base2 = os.path.basename(warp).split('.')[0].split('-')[1]
    #         mask_name2 = print_dir+"/"+base2+"_pred.tif"
    #         if not os.path.exists(mask_name2): continue
    #         mask2 = misc.load_image(mask_name2)
    #         with open(warp, 'rb') as re:
    #             [warp_matrix, cc, overlap] = pickle.load(re)
    #         mask2_warped = cv2.warpAffine(mask2[0], warp_matrix, dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).reshape(orig_shape[1:])
    #         if mrfden == 0:
    #             mrfsum = mask2_warped
    #         else:
    #             mrfsum += mask2_warped
    #         mrfden += 1

    #     if mrfden > 0:
    #         # mrfsum = np.divide(mrfsum, mrfden)
    #         # pred = alpha*pred + (1-alpha) *mrfsum
    #         pred = pred* np.exp(-0.1* (1-mrfsum/mrfden))
    #         fnpred = os.path.join(print_dir, '{}_pred+.tif'.format(base))
    #         scipy.misc.imsave(fnpred, mrfsum.reshape(c.height, c.width) * np.float32(255))

    #     pred = test.postprocess(pred)

    #     val_err += dice_predict(pred, tgt)
    #     val_batches += 1
    #     mod = math.floor(val_batches / num_batches * 10) 
    #     if mod > premod and mod > 0:
    #         print(str(mod*10)+'%..',end="",flush=True)
    #     premod = mod

    # val_error = val_err / val_batches
    # print("Prediction took {:.3f}s".format(time.clock() - start_time))
    # print("  validation loss:\t\t{:.6f}".format(val_error))




