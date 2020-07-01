# Python Script for Extracting Deep Learning Features from Images

This script extracts well-known Deep Learning features from Images using the models and weights in keras. Please see the keras documentation
for details on model architectures, data and training procedures used.

Currently, the script has support for VGG16, ResNet50V2 and MobileNetV2 and can be easily extended to extract further networks in keras.

# Sample usage

## Downloading pre-requisite packages

Please use the requirements.txt file to install the required dependencies. It can be done with:

```
$ pip install -r requirements.txt
```

It is extremely recommended that you use a virtual environment to execute the line above. Conda also works fine.

## Usage Example

Consider a dataset stored in the following file structure:

```
LMD/
   -> f1/
      -> 0001.bmp
      -> 0002.bmp          
      -> ...
      -> 0300.bmp                    
   -> f2/
      -> 0301.bmp
      -> 0302.bmp          
      -> ...
      -> 0600.bmp                          
   -> f3/
      -> 0601.bmp
      -> 0602.bmp          
      -> ...
      -> 0900.bmp
output/
             
```

To output a single numpy array **for each fold** with shape `(n_patches * n_images, n_features)` in the `output` folder:

```
$ python extract_pretrained_features.py --model vgg16 --patches 3 --patches 5 --folds 3 --height 513 --width 1599 -i LMD/f%d/*.bmp -o output/fold-%d_patches-%d.npy
```

This will extract `VGG16` features from the `bmp` image files in the folders `LMD/f1`, `LMD/f2`, and `LMD/f3`. All images will be sliced into `3` and `5` non-overlapping patches. All images must have the same
size and should be specified with the `--height` and `--width` options.

The resulting structure of the `output` folder will be:

```
output/
   -> fold-1_patches-3.npy
   -> fold-2_patches-3.npy
   -> fold-3_patches-3.npy                
   -> fold-1_patches-5.npy
   -> fold-2_patches-5.npy
   -> fold-3_patches-5.npy                   
```

where `fold-X_patches-Y.npy` has the features for fold `X`, with `Y` patches per image. Each file is just a numpy array, with a row per patch. The features are appended to the array according to the lexicographical sorting of the input filenames. The features for the Y patches are placed into consecutive rows. Thus, in the case of `fold-1_patches-3.npy`, the numpy array will contain:

```
[ features for f1/0001.bmp, patch 1,
  features for f1/0001.bmp, patch 2,
  features for f1/0001.bmp, patch 3,
  features for f1/0002.bmp, patch 1,
  features for f1/0002.bmp, patch 2,
  features for f1/0002.bmp, patch 3,
  ...
  features for f1/0300.bmp, patch 1,
  features for f1/0300.bmp, patch 2,
  features for f1/0300.bmp, patch 3 ]
```

