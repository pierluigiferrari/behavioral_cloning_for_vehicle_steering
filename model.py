#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:37:26 2017

@author: pierluigiferrari
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from keras import backend as K
from copy import deepcopy
from PIL import Image
import csv
import cv2
import random

### First, define some helper functions

# Define a function to assemble the training data list to be fed to the generator below.

def assemble_filelists(path='./data/driving_log.csv',
                       angle_adjust=0.15,
                       small_angle_keep=1,
                       keep_threshold=0.05):
    '''
    Assemble a list of training data examples and a list of the corresponding
    labels, both as numpy arrays.
    
    Args:
        path (string, optional): The filepath to a CSV file containing relative
            file names of parallel image data from three cameras, the
            corresponding steering angles of the center camera, throttle,
            brake, and speed, in this order. Throttle, brake, and speed are
            optional. Defaults to './data/driving_log.csv'.
        angle_adjust (float, optional): An offset to add to the steering angles
            of images from the left camera and to subtract from the steering
            angles of images from the right camera. Defaults to 0.15.
        small_angle_keep (float, optional): Must be in [0,1]. Determines what
            percentage of samples with abs(steering_angle) <= threshold to keep
            at random. Defaults to 1.
        keep_threshold (float, optional): Images with a corresponding steering
            angle below `keep_threshold` will be partially and randomly sorted
            out with a ratio of `small_angle_keep`. Defaults to 0.05.
    
    Returns:
        Two numpy arrays. The first has shape (#images,) and contains the list
        of strings with the relative file paths of the image data. Note that it
        does not contain the image data itself. The second array has shape
        (#images, 2), where the first column contains the steering angles of
        the corresponding images and the second column contains the original
        center camera steering angle. The reason for this is that the original
        center camera steering angle is an indicator for the curvature of the
        road, while the adjusted steering angles are not. This information can
        be useful for data augmentation.
    '''
    
    image_files_center = []
    image_files_left = []
    image_files_right = []
    steering_angles_center = []

    with open(path, newline='') as csvfile:
        csvread = csv.reader(csvfile, delimiter=',')
        for i in csvread:
            angle = float(i[3].strip())
            if abs(angle) <= keep_threshold:
                p = np.random.uniform(0,1)
                if small_angle_keep >= p:
                    steering_angles_center.append([angle, angle])
                    image_files_center.append(i[0].strip())
                    image_files_left.append(i[1].strip())
                    image_files_right.append(i[2].strip())
            else:
                steering_angles_center.append([angle, angle])
                image_files_center.append(i[0].strip())
                image_files_left.append(i[1].strip())
                image_files_right.append(i[2].strip())

    assert (len(steering_angles_center) == len(image_files_center) 
            == len(image_files_left) == len(image_files_right))

    image_files_center = np.array(image_files_center)
    image_files_left = np.array(image_files_left)
    image_files_right = np.array(image_files_right)
    steering_angles_center = np.array(steering_angles_center)

    # Produce steering angle values for left and right camera inputs

    steering_angles_left = np.copy(steering_angles_center)
    steering_angles_right = np.copy(steering_angles_center)
    steering_angles_left[:,0] += angle_adjust
    steering_angles_right[:,0] -= angle_adjust

    # Concatenate the inputs from all three cameras to one dataset

    image_files = np.concatenate((image_files_center,
                                  image_files_left,
                                  image_files_right))

    steering_angles = np.concatenate((steering_angles_center,
                                      steering_angles_left,
                                      steering_angles_right))
    
    return image_files, steering_angles

# Define image processing functions to perform the following image
# transformations:
# - Transform the perspective to simulate an incline change
# - Transform the perspective to simulate a curvature change
# - Rotate
# - Translate
# - Flip
# - Scale
# - Change the brightness
# - Histogram-equalize

def transform_incline(image, shift=(5,20), orientation='rand'):
    
    rows,cols,ch = image.shape
    
    hshift = np.random.randint(shift[0],shift[1]+1)
    vshift = hshift
    
    if orientation == 'rand':
        orientation = random.choice(['down', 'up'])
    
    if orientation == 'up':
        hshift = -hshift
        vshift = -vshift
    elif orientation != 'down':
        raise ValueError("No or unknown orientation given. Possible values are 'up' and 'down'.")
    
    pts1 = np.float32([[70,70],
                       [250,70],
                       [0,rows],
                       [cols,rows]])
    pts2 = np.float32([[70+hshift,70+vshift],
                       [250-hshift,70+vshift],
                       [0,rows],
                       [cols,rows]])
    
    #Calculate the transformation matrix, perform the transformation,
    #and return it.
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows))

def transform_curvature(image, shift=(5,30), orientation='rand'):
    
    rows,cols,ch = image.shape
    
    shift = np.random.randint(shift[0],shift[1]+1)
    
    if orientation == 'rand':
        orientation = random.choice(['left', 'right'])
    
    if orientation == 'left':
        shift = -shift
    elif orientation != 'right':
        raise ValueError("No or unknown orientation given. Possible values are 'left' and 'right'.")
    
    pts1 = np.float32([[70,70],[250,70],[0,rows],[cols,rows]])
    pts2 = np.float32([[70+shift,70],[250+shift,70],[0,rows],[cols,rows]])
    
    #Calculate the transformation matrix, perform the transformation,
    #and return it.
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows)), shift

def do_rotate(image, min=5, max=15, orientation='rand'):
    
    rows,cols,ch = image.shape
    
    #Randomly select a rotation angle from the range passed.
    random_rot = np.random.randint(min, max+1)
    
    if orientation == 'rand':
        rotation_angle = random.choice([-random_rot, random_rot])
    elif orientation == 'left':
        rotation_angle = random_rot
    elif orientation == 'right':
        rotation_angle = -random_rot
    else:
        raise ValueError("Orientation is optional and can only be 'left' or 'right'.")
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2), rotation_angle, 1)
    return cv2.warpAffine(image, M, (cols, rows)), -rotation_angle

def do_translate(image, horizontal=(0,40), vertical=(0,10)):
    
    rows,cols,ch = image.shape
    
    x = np.random.randint(horizontal[0], horizontal[1]+1)
    y = np.random.randint(vertical[0], vertical[1]+1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])
    
    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift

def do_flip(image, orientation='horizontal'):
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)

def do_scale(image, min=0.9, max=1.1):
    
    rows,cols,ch = image.shape
    
    #Randomly select a rotation angle from the range passed.
    scale = np.random.uniform(min, max)
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows))

def change_brightness(image, min=0.5, max=2.0):
    
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)
    
    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2]*random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2]*random_br)
    hsv[:,:,2] = v_channel
    
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def histogram_eq(image):
    
    image1 = np.copy(image)
    
    image1[:,:,0] = cv2.equalizeHist(image1[:,:,0])
    image1[:,:,1] = cv2.equalizeHist(image1[:,:,1])
    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])
    
    return image1

# Define a generator function that can do data conversion and augmentation.

def generate_batch(filenames,
                   labels,
                   batch_size=128,
                   resize=False,
                   gray=False,
                   equalize=False,
                   brightness=False,
                   flip=False,
                   incline=False,
                   curvature=False,
                   curve_correct=0.008,
                   rotate=False,
                   rot_correct=0.03,
                   translate=False,
                   trans_correct=0.003):
    '''
    Generate batches of samples and corresponding labels indefinitely from
    lists of filenames and labels.
    
    Yields two numpy arrays, one containing the next `batch_size` samples
    from `filenames`, the other containing the corresponding labels from
    `labels`.
    
    Shuffles `filenames` and `labels` consistently after each complete pass.
    
    Can perform image transformations for data conversion and data
    augmentation. `resize`, `gray`, and `equalize` are image conversion tools
    and should be used consistently during training and inference.
    The remaining transformations serve for data augmentation specifically for
    the task of predicting steering angles for driving from image data and
    should only be used during training. Each data augmentation process can set
    its own independent application probability.
    
    `prob`, `mode`, and `threshold` work the same in all arguments in which
    they appear:
    
    `prob` must be a float in [0,1] and determines the probability that the
    respective transform is applied to any given image.
    
    `mode` must be an integer in {0,1,2,3} and determines how the application
    of the respective transform is conditional on the image's original center
    camera steering angle.
        mode 0: The application of the transform does not depend on the image's
                steering angle. `threshold` does not need to be specified.
        mode 1: If `threshold` > 0: Apply transform only to images with
                `angle` >= `threshold`.
                If `threshold` < 0: Apply transform only to images with
                `angle` <= `threshold`.
        mode 2: Apply transform only to images with `abs(angle)` >= `threshold`.
                `threshold` should be positive.
        mode 3: Apply transform only to images with `abs(angle)` <= `threshold`.
                `threshold` should be positive.
    
    `threshold` must be a scalar in the same range as `labels` and determines
    the threshold to trigger the application of the respective transform
    depending on `mode`.
    
    Note that for a given image and a given transform, the generator first
    tests whether the threshold criterion is met and *afterwards* determines
    based on `prob` whether the transform will be applied to the image. This
    means that in all modes other than 0, `prob` determines the ratio of images
    to which the respecitve transform is applied out of all *eligible* images,
    not out of all images.
    
    All conversions and transforms default to `False`.
    
    Args:
        filenames (array-like): A 1-D list or numpy array containing all file
            paths to the samples from which the batches are to be generated.
            Note that `filenames` must not contain the actual files themselves.
        labels (array-like): A list or numpy array containing the labels
            corresponding to the samples. Important note: Label lists are now
            expected to contain tuples of two floats for each sample for some
            data augmentation processes to work (all those that alter the
            image's steering angle), rather than just one scalar value as you
            might expect. The first float of the tuple is expected to be the
            image's steering angle, the second is expected to be the original
            center camera steering angle for the respective image. The reason
            for this is that the original center camera steering angle is an
            indicator for the curvature of the road, while the adjusted side
            camera steering angles are not. This information is useful for a
            more targeted application of some transforms.
        batch_size (int, optional): The size of the batches to be generated.
            Defaults to 128.
        resize (tuple, optional): A 1-D tuple of 2 integers for the desired
            output size of the images in pixels. The expected format is
            (width, height).
        gray (bool, optional): If `True`, converts the images to grayscale.
        equalize (bool, optional): If `True`, performs histogram equalization
            on the images. This can improve contrast and lead the improved
            model performance.
        brightness (tuple, optional): `False` or a tuple containing three
            floats, (min, max, prob). The brightness of the image will be
            scaled by a factor randomly picked from a uniform distribution in
            the boundaries of [min,max]. Both min and max must be >=0.
        flip (float, optional): `False` or a float in [0,1], see `prob` above.
            Flip the image horizontally. Also inverts (additive inverse) the
            respective steering angle.
        incline (tuple, optional): `False` or a tuple of five elements:
            (min, max, prob, mode, threshold). Transforms the perspective of
            the image randomly to simulate either an increase or decrease of
            the incline of the road. `min` and `max` determine the strength of
            the incline change and must be integers in [0,50], although
            reasonable values should lie within [0,30].
        curvature (tuple, optional): `False` or a tuple of five elements:
            (min, max, prob, mode, threshold). Transforms the perspective of
            the image randomly to simulate a change of the curvature to the
            left or to the right. `min` and `max` determine the strength of the
            curvature change and must be integers in [0,50], although
            reasonable values should lie in [0,30]. The steering angle of the
            image is adjusted accordingly, see the next argument. Further
            information: A randomly picked value from [min, max] determines the
            horizontal shift of an imaginary horizontal line that marks the
            separation between the road and the horizon. The shift in the
            curvature is achieved by shifting this line to the left or to the
            right and perspectively transforming all other pixels in the image
            along with it.
        curve_correct (float, optional): Must be non-negative. The amount by
            which the steering angle is corrected per pixel of horizontal
            curvature change. Computed as follows:
            Steering angle += curve_correct * curvature_shift. Defaults to 0.01.
        rotate (tuple, optional): `False` or a tuple of five elements:
            (min, max, prob, mode, threshold). Rotate the image randomly
            clockwise or counter-clockwise. The rotation angle is picked from
            [min, max], where `min` and `max` must be integeres and determine
            the rotation angle in degrees. The orientation of the rotation is
            random. The respective steering angle of the image is corrected
            accordingly, see the next argument. Defaults to `False`.
        rot_correct (float, optional): Must be non-negative. The amount by
            which the steering angle is corrected per degree of rotation of the
            image. Computed as follows:
            Steering angle += rot_correct * rotation_angle. Defaults to 0.03.
        translate (tuple, optional): `False` or a tuple, with the first two
            elements tuples containing two integers each, and the third element
            a float: ((min, max), (min, max), prob). The first tuple provides
            the range in pixels for horizontal shifts of the image, the second
            tuple for vertical shifts. The number of pixels to shift the image
            by is uniformly distributed within the boundaries of [min,max],
            i.e. `min` is the number of pixels by which the image is translated
            at least. Both min and max must be >=0. The steering angle
            corresponding to the image is adjusted for horizontal translation
            (but not for vertical translation), see the subsequent argument.
            Detaults to `False`.
        trans_correct (float, optional): Must be non-negative. The amount by
            which the steering angle is corrected per pixel of horizontal
            translation. Computed as follows:
            Steering angle += trans_correct * horizontal_shift.
            Defaults to 0.004.
    
    Yields:
        The next batch as a 1-D tuple containing two numpy arrays. The first
        contains the batch samples as numpy arrays, the second contains the
        corresponding labels.
    '''
    
    assert len(filenames) == len(labels), "The lengths of `filenames` and `labels` must be equal."
    assert not (len(filenames) == 0 or filenames is None), "`filenames` cannot be empty."
    
    current = 0
    
    while True:
        
        batch_X, batch_y = [], []
        
        #Shuffle the data after each complete pass
        if current >= len(filenames):
            filenames, labels = shuffle(filenames, labels)
            current = 0
        
        for filename in filenames[current:current+batch_size]:
            with Image.open(filename) as img:
            #with Image.open('./data/{}'.format(filename)) as img:
                batch_X.append(np.array(img))
        batch_y = deepcopy(labels[current:current+batch_size])
        current += batch_size
        
        #At this point we're done producing the batch. Now perform some
        #optional image transformations:
        
        if equalize:
            batch_X = [histogram_eq(img) for img in batch_X]
        
        if brightness:
            for i in range(len(batch_X)):
                p = np.random.uniform(0,1)
                if p >= (1-brightness[2]):
                    batch_X[i] = change_brightness(batch_X[i],
                                                   min=brightness[0],
                                                   max=brightness[1])

        if flip:
            if flip[1] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-flip[0]):
                        batch_X[i] = do_flip(batch_X[i])
                        batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 1 and flip[2] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i] >= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 1 and flip[2] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i] <= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i]) >= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i]) <= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(flip[1]))
        
        if incline:
            if incline[3] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-incline[2]):
                        batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 1 and incline[4] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] >= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 1 and incline[4] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] <= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) >= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) <= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(incline[3]))
            
                    
        if curvature:
            if curvature[3] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-curvature[2]):
                        if batch_y[i,1] > 0:
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='right')
                            batch_y[i,0] += curve_correct * cshift
                        elif batch_y[i,1] < 0:
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='left')
                            batch_y[i,0] += curve_correct * cshift
                        else:
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='rand')
                            batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 1 and curvature[4] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] >= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='right')
                            batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 1 and curvature[4] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] <= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='right')
                            batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) >= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            if batch_y[i,1] > 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='right')
                                batch_y[i,0] += curve_correct * cshift
                            elif batch_y[i,1] < 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='left')
                                batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) <= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            if batch_y[i,1] > 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='right')
                                batch_y[i,0] += curve_correct * cshift
                            elif batch_y[i,1] < 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='left')
                                batch_y[i,0] += curve_correct * cshift
                            else:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='rand')
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(curvature[3]))
        
        if rotate:
            if rotate[3] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-rotate[2]):
                        batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                        batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 1 and rotate[4] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] >= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 1 and rotate[4] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] <= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) >= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) <= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(rotate[3]))
            
        if translate:
            for i in range(len(batch_X)):
                p = np.random.uniform(0,1)
                if p >= (1-translate[2]):
                    batch_X[i], hshift = do_translate(batch_X[i], translate[0], translate[1])
                    batch_y[i,0] += trans_correct * hshift
        
        if resize:
            batch_X = [cv2.resize(img, dsize=resize) for img in batch_X]
            
        if gray:
            batch_X = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                                                 for img in batch_X]), 3)
        
        yield (np.array(batch_X), np.array(batch_y[:,0]))

# Define a function to build the model

def build_model():
    
    # Input image format
    in_row, in_col, ch = 80, 160, 3
    
    # Cropping
    cr_lef, cr_rig, cr_top, cr_bot = 5, 5, 20, 10 # How many pixels to be cropped in each direction
    cr_row, cr_col = in_row-cr_top-cr_bot, in_col-cr_lef-cr_rig # The resulting cropped image format
    
    model = Sequential()
    
    model.add(Cropping2D(cropping=((cr_top, cr_bot), (cr_lef, cr_rig)),
                         input_shape=(in_row, in_col, ch)))
    model.add(Lambda(lambda x: x/127.5 - 1., #Convert input feature range to [-1,1]
                     output_shape=(cr_row, cr_col, ch)))
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode="valid"))
    model.add(BatchNormalization(axis=3, momentum=0.99))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(BatchNormalization(axis=3, momentum=0.99))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, subsample=(2, 2), border_mode="valid"))
    model.add(BatchNormalization(axis=3, momentum=0.99))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(BatchNormalization(axis=1, momentum=0.99))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)
    
    model.compile(optimizer=adam, loss="mse")
    
    return model

### Load data and perform training

image_files, steering_angles = assemble_filelists()

# Split off a validation dataset from the data and shuffle
X_train, X_val, y_train, y_val = train_test_split(image_files,
                                                  steering_angles,
                                                  test_size=0.2)

batch_size = 128
epochs = 1

train_generator = generate_batch(X_train,
                                 y_train,
                                 batch_size=batch_size,
                                 resize=(160,80),
                                 brightness=(0.4, 1.1, 0.1),
                                 flip=(0.5, 0),
                                 curvature=(5, 30, 0.5, 0),
                                 curve_correct=0.008,
                                 translate=((0, 40), (0, 10), 0.5),
                                 trans_correct=0.003)

val_generator = generate_batch(X_val,
                               y_val,
                               batch_size=batch_size,
                               resize=(160,80))

#Clear previous models from memory.
K.clear_session()

model = build_model()
#model.load_weights('./model_15_weights.h5')

#model = load_model('./model_15.h5')

history = model.fit_generator(generator = train_generator,
                              samples_per_epoch = len(y_train),
                              nb_epoch = epochs,
                              callbacks = [EarlyStopping(monitor='val_loss',
                                                         min_delta=0.0001,
                                                         patience=2),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.2,
                                                             patience=1,
                                                             epsilon=0.0001,
                                                             cooldown=0)],
                              validation_data = val_generator,
                              nb_val_samples = len(y_val))

model_name = 'model_16'
model.save('./{}.h5'.format(model_name))
model.save_weights('./{}_weights.h5'.format(model_name))

print()
print("Model saved as {}.h5".format(model_name))
print("Weights also saved separately as {}_weights.h5".format(model_name))
print()
