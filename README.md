## Deep learning CNN toolkit 

This toolkit was used for the Kaggle Ultrasound nerve segmentation challenge.

## Instructions
### Data
The training data need to be stored in a 2D image format (e.g. png, tif) inside a train directory.

- For segmentation the train directory has to contain a mask for each image:  
  if the image is stored as train/image1.tif the mask will be train/image1_mask.tif
  
- For autoencoders only the images will be inside the train directory (no masks).

The testing data are stored as images inside a test directory (no masks).

### Training

Training is performed with the train.py with parameters:  
<table>
<tr><td>-v [version] : </td><td> version of the experiment to be run (see below for configuration of the experiment)</td></tr>
<tr><td>-train [dir] : </td><td> train directory (default: train)  
</td></tr><tr><td>-cv [num_folds] : </td><td> cross-validation with a number of folds (default: 10)  
</td></tr><tr><td>-fold [fold_nr] : </td><td> run for a specific fold of the cross-validation  
</td></tr><tr><td>-seed [seed_nr] : </td><td> change the seed used for the cross-validation (default: 1234)  
</td></tr></table>

### Testing

Testing is performed with the test.py. Test.py has the same parameters as above (apart from -train) and additionally
<table>
<tr><td>-test [dir] : </td><td> test directory (default: test)  
</td></tr><tr><td>-results [dir] : </td><td> results directory (default: submit)  
</td></tr></table>

Additional parameters can be viewed by running the test.py with the --help option.

### Configuration

The config_default.py contains the default parameters used for training.  
This can be overriden for the different experiments if required.  
In order to setup a new experiment the user needs to:  
- create a directory with the experiment name inside the "params" folder (e.g. params/unet-small)
- write a config.py file inside the experiment directory (e.g. params/unet-small/config.py) with the parameters he wants overriden (e.g. depth=4). Parameters that are not specified retain their original values from the config_default.py
- run the training/testing specifying the experiment as the version (e.g. python train.py -v unet-small )
