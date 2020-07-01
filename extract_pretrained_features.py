#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras.applications.vgg16 as vgg16
import keras.applications.resnet_v2 as resnetv2
import keras.applications.mobilenet_v2 as mobilenetv2
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import click
 
def next_patch(spec, n):
    step = math.floor(spec.shape[1] / n)
    for i in range(n):
        yield spec[:,i*step:(i+1)*step,:]

def get_model(model, **kwargs):
    if model == 'vgg16':
        return vgg16.VGG16(**kwargs), vgg16.preprocess_input
    if model == 'resnet50v2':
        return resnetv2.ResNet50V2(**kwargs), resnetv2.preprocess_input
    if model == 'mobilenetv2':
        return mobilenetv2.MobileNetV2(**kwargs), mobilenetv2.preprocess_input
    
    raise ValueError

def extract_features(model_name, patches, folds, spec_height, spec_width, 
    input_path_proto, output_proto, gpuid):

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuid)
    for n_patches in patches:
        print("Slicing images into %d non-overlapping patches..." % (n_patches))
        K.clear_session()

        input_shape = (spec_height, math.floor(spec_width/n_patches), 3)
        
        model, preprocess_input = get_model(model_name, weights='imagenet', include_top=False, 
                          input_shape=input_shape, pooling='avg')

        for fold in folds:
            print("Extracting features for fold %d..." % (fold))
            if len(glob.glob(input_path_proto % (fold))) == 0:
                raise RuntimeError("No files found in: %s" % (input_path_proto % (fold)))

            features = []
            for fname in sorted(glob.glob(input_path_proto % (fold))):
                img = image.load_img(fname)
                spec = image.img_to_array(img)
                for p in next_patch(spec, n_patches):
                    p = preprocess_input(p)
                    p = np.expand_dims(p, axis=0)
                    features.append(model.predict(p))

            features = np.concatenate(features)
            output_filename = output_proto % (fold, n_patches)
            np.save(output_filename, features)
            print(output_filename, features.shape)

@click.command()
@click.option(
    '--model',
    type=click.Choice(['vgg16', 'resnet50v2', 'mobilenetv2']),
    help='Which pretrained net to use for feature extraction',
    required=True
)
@click.option(
    '--patches',
    '-p',
    multiple=True,
    default=[1],
    help='Into how many patches should the image be sliced into',
    type=int
)
@click.option(
    '--folds',
    default=1,
    help='How many cross-validation folds to consider',
    type=int
)
@click.option(
    '--height',
    required=True,
    help='Height of the input images',
    type=int
)
@click.option(
    '--width',
    required=True,
    help='Width of the input images',
    type=int
)
@click.option(
    '--input-file-format-string',
    '-i',
    required=True,
    help='A format string containing the full path to the input images. A single %d argument indicates' +
        " the position of the fold number. Example: f%d/*.bmp would match all bmp files  in each f1, f2, ..., fn fold folders."
)
@click.option(
    '--output-file-format-string',
    '-o',
    required=True,
    help='A format string containing the full path to the output. There should be exactly two ' +
        '%d format arguments, the first one indicates the position of the fold number and the second ' +
        '%d indiates the position of the patch number. Example: output/fold-%d_patch-%d.npy ' +
        'saves the output in the output folder, where the features of each patch is saved into a ' +
        'file called fold-%d_patch-%d.npy.'
)
@click.option(
    '--gpuid',
    '-g',
    default=0,
    help='The GPUID of the GPU to use. (check available GPUs with nvidia-smi)',
    type=int
)
def main(model, patches, folds, height, width, input_file_format_string, output_file_format_string,gpuid):
    folds = list(range(1,folds+1))
    patches = list(patches)
    print("Feature Extraction Parameters")
    print("Pre-trained model: %s" % (model))
    print("Non-overlapping patches per image: %s" % str(patches))
    print("Folds: %s" % str(folds))
    print("Image Dimensions h=%d, w=%d " % (height, width))
    print("Format string for input: %s " % (input_file_format_string))
    print("Format string for output: %s " % (output_file_format_string))
    print("GPU ID: %d" % (gpuid))
    print("")

    extract_features(model, patches, folds, height, width, 
        input_file_format_string, output_file_format_string, gpuid)

if __name__ == "__main__":
    main()
