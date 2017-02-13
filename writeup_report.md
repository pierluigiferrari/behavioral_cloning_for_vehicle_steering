**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity-provided simulator and my drive.py file, the car can be driven autonomously around both the lake and jungle tracks by executing
```sh
python drive.py model.h5
```
My drive.py is slightly modified to perform input image resizing and to vary the throttle to limit the speed between min=12 mph and max=24 mph.

####3. Submssion code is usable and readable

The part of the model.py file that contains the code for creating, training and saving the model starts in line 643. The lines before define a number of helper functions that are part of the overall training pipeline: An `assemble_filelists()` function to assemble lists of the available training data from the drive_log.csv, a `generate_batch()` generator function used by Keras' `fit_generator()` function to train the model, and a bunch of image transformation functions that are used by the generator to do ad-hoc data augmentation during training. These helper functions are documented in detail where relevant.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The architecture of my model is as follows (lines 643-680 of model.py):
- Color image input with dimensions 80x160x3
- Keras Cropping2D layer to crop the input to 50x150 pixels
- Keras Lambda layer to convert the feature value range to [-1,1]
- Three convolutional layers with depths 32, 64, and 128, filter sizes 8x8, 5x5, and 3x3, and strides 4, 2, and 2.
- One dense layer with 512 units following the conv layers, and one output unit
- ELUs as nonlinearities after each layer except the output unit
- Batch normalization after each layer except the output unit
- Dropout after the second and third conv layers (both rates 0.3) and after the first dense layer (rate 0.5)

####2. Attempts to reduce overfitting in the model

As described above, the model performs batch normalization after each layer, which helps reduce overfitting. In addition to that, the three dropout layers above were implemented. I did not encounter signs of overfitting in general, even without the dropout layers.

The model was validated on a held-out validation dataset that it hadn't seen during training to ensure that the model was not overfitting (line 687 of model.py). Every iteration of the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer with a default learning rate of 0.001 and a decay of 5e-05, so the learning rate was not tuned manually (line 676 of model.py). In addition to that, early stopping and learning rate reduction mechanisms in case of the validation loss plateauing for several epochs in a row, see the callbacks starting in line 721 of model.py.

####4. Appropriate training data

I recorded training data exclusively from good human driving behavior, i.e. following the lane trajectory as cleanly as possible, trying to stay centered on the lane as well as I could and taking curves as smoothly as I could. I recorded no recovery sequences whatsoever, i.e. no sequences where I drove from the edges of the lane back towards the center. My reasoning was that if real cars on the road don't have that luxury of recording recovery data, then I shouldn't need such data either in order to train a good model.

For details about how I created and augmented the training data, see the last section of this report.

###Architecture and Training Documentation

####1. Solution Design Approach

Initially I trained two different promising architectures to see if there would be a significant difference in performance between the two. The first was the architecture from NVIDIA's 2016 paper ["End to End Learning for Self-Driving Cars"](https://arxiv.org/abs/1604.07316), albeit with different input dimensions of 50x150 pixels, the second was a modified version of [this](https://github.com/commaai/research/blob/master/train_steering_model.py) Comma.ai model.

I decided that if these two architectures would perform relatively similarly and if both would be capable of training models that manage to stay on the lake track, then I would not expand my architecture search, because it would indicate that good results are likely possible with many different architectures and that the collection and augmentation of the training data would likely have a far more significant influence on the training success than the specific network architecture chosen.

I was not able to measure a significant difference in performance between the two models, and I was able to get good results with both, so I stuck with the latter model - you can find the architecture details in the section above. While model architecture is important, I feel as long as you pick one of many possible suitable architectures, the kind of data you collect and the kind of data augmentation you perform are the much more crucial and more difficult parts to solve this problem well. In other words, the model architecture you choose likely won't be the limiting factor here - the collected data and data augmentation techniques will be.

Hence, the majority of my work (and of my code, as you can see in model.py) focused on the collection and processing of the training data. Please find the details on this in the last section of this report.

I split my collected data into training and validation datasets using Scikit-Learn's train_test_split(), although I quickly understood that a low validation error is not necessarily indicative of a good model that can stay on the track, because it represents only a small fraction of the spectrum of possible situations, and failing in even just one situation might be enough for your model not to make it around the whole track. It can happen that you get a really low validation error but your car still misses a difficult turn. The other thing is that, even if the correct behavior for that turn is reflected in the validation data, the difference in the validation error between your model making or missing that difficult turn, all else being equal, will be tiny. The bottom line is that I find the validation error helpful to prevent overfitting and to make sure that your model is roughly evolving in the right direction, but overall it was not the relevant metric I looked at during training - to be honest, how the validation error changed played almost no role in my decisions on next steps to take. The main criterion I focused on was putting each iteration of the model to the test in the autonomous mode of the simulator, everything else is just semi-useful proxies.

The most important factor to train a successful model was what kind of data processing to perform in order to get a model that can generalize to recover from as many different suboptimal lane positions as possible. As already mentioned, please find the details on this in the last section.

My network architecture remained almost unchanged throughout the iterations of my training process. Only occasionally I would train the NVIDIA architecture with the exact same data processing and learning parameters as my main architecture as a safety check to validate that the different architectures between the two models did not cause a measurable difference in performance. Other than this occasional architecture validation, I stuck with the architecture described above throughout the entire project.

Throughout my training, unsuccessful models differed from successful models only by the kind of data they were trained on, and thus the main focus lay on generating the most ideal input data.

My procedure was to get a model that works partially on the lake track and then iteratively improve on that model until it would eventually be able to master the jungle track, too. Please find more on the training procedure below.

The final model that was submitted here is able to run laps on both the lake track and the jungle track. My submitted drive.py limits the car's speed to 24 mph, but this limitation was merely chosen to make the car's driving behavior a bit smoother on the jungle track. You could remove the speed limit altogether - at a throttle of 0.2 the model was still able to run multiple laps on the jungle track.

####2. Final Model Architecture

The final model architecture is described in the model architecture section above. The respective lines in model.py are lines 643-680.

Notable details of the architecture are that batch normalization is applied after each convolutional and each dense layer and that ELU nonlinearities were used instead of the more common ReLUs. The latter decision was based on the observation that the small additional computational cost seems to be worth paying in return for not having the function's the gradient be zero anywhere.

Instead of following the conv layers with max pooling layers I used a convolutional stride greater than 1, namely 4 for the first conv layer and 2 for the second and third conv layers. I didn't do this because I had good theoretical reasons for it, but because the Comma.ai model that my model is based on did it this way and I considered it an experiment to see if it would produce good results. Intuitively, reducing complexity through pooling should be superior over reducing complexity by increasing the stride in the conv layers, since the former method chooses the most relevant information out of all possible filter positions, while the latter method loses information by skipping many filter positions to begin with. In practice, however, it seems to work well enough.

In research papers I read I noticed a trend towards using fewer fully connected layers (if any at all) as they make up the majority of the computational complexity and models with few fully connected layers seem to be working well. Indeed, one fully connected layer of 512 seems to be good enough to train a successful model - maybe fewer units in this layer would have sufficed, too.

Here is a visualization of the architecture. The default visualization that comes with Keras is not exactly pretty, but I didn't have the time to figure out how to generate a pretty one without having to manually draw it myself and at least it shows the layer dimensions:

![Model architecture][./model.png]

####3. Creation of the Training Set & Training Process

Collecting a well-balanced mix of good training data and performing the right data augmentation techniques to it was the single most decisive factor that determined the success of the training. While the realization that the training data matters is of course trivial, putting together well-balanced training data in this case is not as trivial, but particularly important. The range of possible steering angles is nearly continuous, so there is a wide spectrum of training data labels to cover in a balanced way.

Data augmentation is particularly difficult in this case, because, as opposed to many classification tasks, for many relevant transformations of the input data, the corresponding labels need to be adjusted, too. Knowing **how** exactly to adjust the steering angle when transforming an image in a certain way turns into a project of its own, and a lot of thought needs to go into it. A bear is still a bear whether you flip the image or not, but the steering angle of the perspectively distorted image of a road might need to be adjusted in a non-obvious way. Further below I will describe my findings regarding data augmentation methods that worked and didn't work, and regarding those that were actually useful and those that turned out to be unnecessary.

First I recorded roughly five laps of good driving behavior in the default direction (counter-clockwise) on the lake track (track 1), followed by roughly two laps in the reverse direction (clock-wise) in order to compensate a bit for the imbalance of left and right curves on the track. I would later flip half of the images horizontally anyway, so that this imbalance theoretically doesn't matter, but I wanted the additional data anyway. I ended up with a little more than 45,000 images when I was done recording, i.e. around 15,000 per camera.

With that data from the lake track only I wanted to get the model to master the lake track and see how far I can get on the mountain and jungle tracks without using any data recorded on those tracks.

With some data augmentation (see details below) it was possible to get the model to drive autonomously on both the lake and mountain tracks, without it ever having seen the mountain track during training, but I was not able to get any useable results on the jungle track without training on any data from it. 4-6 epochs on 90% of the the above dataset (the remaining 10% were held out for validation) were enough to train a model to drive smoothly on the lake track using an Adam optimizer with the default learning rate of 0.001. The use of batch normalization might have enabled accelerated training.

Note that I deliberately did not record any recovery data, i.e. I did not record any data of the car correcting its course from the edges of the lane back towards the center. Since real cars on real roads cannot really make use of this technique and can still learn how to drive autonomously, my model should be able to learn without this data, too. I hoped that the left and right camera images and some geometric transformations of the images would be enough to produce the same effect of recovery data, which turned out to be true. Not to mention that it is a lot more efficient than recording lots of manually produced recovery data.

I used the data from all three cameras for the training. The images from the two non-center cameras turned out to be incredibly useful to train off-center correction. I experimented with different steering angle adjustments for the left and right cameras, ranging from adding/subtracting constants between 0.1 and 0.25. 0.15-0.2 turned out to be reasonable values. I also experimented with non-constant adjustment values that depend on the magnitude of the center camera steering angle, the reasoning being that the larger the curve radius, the more time the car has to revert back towards the center of the lane, allowing for smoother corrections, while the sharper the curve, the faster the car has to revert back toward the center. By contrast, if the angle adjustment is an additive constant, the correction back to the center of the lane is always equally fast (which means equally abrupt), regardless of the curvature of the road. I ended up discarding the magnitude-dependent approach though, since it introduced more complexity for unclear gain.

I reduced the original size of the recorded images (160x320 pixels) by half in both dimensions to 80x160 pixels. I then cropped away the top 20 and the bottom 10 pixels because they just contain the sky and the hood of the car, respectively - visual information that is irrelevant to predict the steering angle. I also cropped away 5 pixels each on the left and right for the same reason. It might be useful to crop even more pixels from the top to eliminate even more irrelevant or even misleading image information, but I got satisfactory results with my processing.

Now about the data augmentation techniques I experimented with. I tested the following:

- Flipping images horizontally to prevent a bias towards being able to handle some situations only in one direction but not the other. The steering angle is being inverted (additive inverse) accordingly.
- Changing the brightness, particularly decreasing it, to make the model less dependent on certain colors, to make it recognize lane markings with less contrast, and to cater to the darker colors of the mountain track.
- Three kinds of transformations came to my mind as possible candidates to correct off-center positions of the car on the lane and to ensure that it can handle sharp curves well: Rotation, horizontal translation, and a perspective transform simulating a change in the curvature of the road. I tested the effectiveness of all three and report my findings below.
- Transforming the perspective to simulate an incline change uphill or downhill. The purpose of this was to use the data from the flat lake track to train the model for the mountain and jungle tracks, both of which contain many slope changes.

Here is an example of some of these transformations. The original image for comparison:

[image1]: ./examples/00_original.png "Original, steering angle == 0.00"

Translated horizontally by 30 pixels:

[image2]: ./examples/01_translate.png "Translated, steering angle == 0.09"

Perspective transform to simulate a left turn / orientation of the car to the right edge of the lane:

[image3]: ./examples/02_curvature.png "Curved to the left, steering angle == -0.32"

Perspective transform to simulate a downhill road:

[image4]: ./examples/03_incline_down.png "Incline downhill, steering angle == 0.00"

Perspective transform to simulate an uphill road:

[image5]: ./examples/04_incline_up.png "Incline uphill, steering angle == 0.00"

Horizontal flip:

[image6]: ./examples/05_flip.png "Flip, steering angle == -0.00"

Results of my data augmentation experiments:

- Horizontal flipping: This one is a no-brainer - unsurprisingly it turned out to be very useful.
- Changing the brightness: It had exactly the desired effect. Thanks to decreasing the brightness of the lake track images, the model was able to drive on the mountain track without ever having seen it during training. Depending on the training iteration, I randomly varied the brightness of 10-50% of the images between factor 0.4 and 1.5 of the original brightness.
- Translation: Horizontal translation is just an extension of the effect of using the left and right camera images and turned out to be very useful, if not essential, to training a model that stays close to the center of the lane. I randomly horizontally translated the images by 0 to 40 pixels, sometimes 0 to 50 pixels, and steering angle adjustments of 0.003-0.004 per pixel of translation turned out to yield reasonable correction speeds that are neither too abrupt on straight roads nor too slow in sharp curves. Vertical translation turned out to be unnecessary. I did it a little bit (0-10 pixels) just to create more diverse data, but vertical translation does not serve as an even remotely realistic proxy for simulating changes in the slope of the road.
- Curvature perspective transform: This turned out to be useful to simulate sharper curves on the one hand, but even more importantly it simulates situations in which the car is oriented at an angle to the lane rather than parallel to the lane. The image above illustrates this effect. If you compare the central vertical gridline in the original image and the distorted image you see that the distorted image simulates the car being oriented toward the side of the lane rather than toward the center of the road as in the original image. Of course, this primitive perspective distortion is a very imperfect proxy for a change in the curvature of the road. To truly increase the sharpness of a curve in a realistic way for example, one can of course not just shift the pixels in the linear way that this transform does, but this approximation still did an alright job. In order to understand the steering angle adjustment factor you would have to read the code, but I documented the generator function in great detail in case you're interested.
- Rotation: I experimented with rotating images to simulate a change in the curvature of the road, but in most cases this does not yield a realistic approximation, and more importantly it is inferior to the perspective transform described above. I did not end up using this transform.
- Incline perspective transform: While it generally actually is a more realistic approximation than the curvature transform above, it turned out to be completely unnecessary - I did not end up using this.

All the transforms above are defined as small helper functions in lines 117-250 of model.py.

The function that actually applies these transformations is the generator function defined in lines 254-639 of model.py. The large number of code lines is mostly owed to distinguishing between different cases triggered by options in the arguments. Feel free to read the documentation, I documented the function in great detail as it is an important part of the training setup. In a nutshell, it loads batches of training data and labels, applies the transforms specified in the arguments, yields the results, shuffles the dataset upon each complete pass, and can do this indefinitely. Each transform has its own independent application probability and some can choose from a number of modes to operate in - see the documentation.

The generator function provides some options to apply the above image transforms in a more targeted way. For example, for the curvature transformation, the `mode` argument specifies whether all images are eligible for the transform, or only images with a certain minimum or maximum corresponding absolute steering angle, or only to images with a corresponding steering angle that is positive or negative. During training, it sometimes proved helpful to apply the curvature transform only to images of an already curved road, and even more than that it was sometimes helpful to apply the artificial curvature only in the same direction as the original curvature. One possible reason for the latter phenomenon might be that the steering angle adjustment associated with the artificial curvature change is not chosen perfectly, and if a road that was curved to the left is artificially straightened by being transformed to the right does not end up with the appropriate steering angle (e.g. zero), then this might create conflicting training data for the model. Note that the assemble_filelists() function returns the steering angle list as a list with two columns, containing not only the steering angle for an image, but also the original steering angle of the center camera version of the respective image. The reason for this is that the original center camera steering angle is a reasonable indicator for the actual curvature of the road (assuming that I drove relatively cleanly along the trajectory of the road) while the adjusted steering angles of the left and right camera images are not. Example: If an image has a steering angle of -0.15, it might be a slight left turn, but it might also be the right camera image of a straight part of the road (or neither). Hence it is useful to preserve the original steering angle associated with the center camera image for all images. The `mode` option in the generator function uses this original center camera steering angle to decide which images are eligible for transformation an which aren't.

As mentioned above and unsurprisingly, trying to get the model to work on the jungle track while having trained it only on the lake track was unsuccessful. However, even **after** training it on jungle track data I initially had difficulties getting it to drive on the track correctly. I anticipated that sharp turns would be an issue, but those didn't cause any problems. There were three other leading causes of failure in my training results - see the images below. The first were sudden downhill parts where a sharp edge marks the end of the visible road before it goes downhill and the car cannot see early enough what lies ahead. This problem was sometimes exacerbated by the second leading cause of failure, unrelated road stretches on the horizon creating the optical illusion of being the continuations of the road the car is currently on, leading the model to follow the road stretch on the horizon rather than the road it was on. The third leading cause of failure were the two directly adjacent road stretches at the start of the track. The road is completely straight there, but my model initially still had difficulties staying straight, it always wanted to pull over to the other side of the road. It took recording a bunch of extra data on this stretch at the start to get this problem under control.

Here are illustrations of these difficulties:

![Sudden downhill part][./issues/issue02_downhill.png]

The car can't see what lies ahead - and it's a sharp right turn. Exacerbating this, the left lane marking of the road is continued by the left lane marking of the distant road stretch on the horizon. The model might take this as a cue to drive straight.

[Misleading road stretch on the horizon][./issues/issue03_misleading.png]

The road makes a sharp right turn, but there is also a straight stretch on the horizon, creating the illusion of a fork in the road.

[Adjacent roads][./issues/issue04_adjacent.png]

These two adjacent road stretches have nothing to do with each other, but the model had a lot of difficulties to tell which is the correct one to drive on. Initially it constantly tried to pull over to the other side of the road.

I ended up recording around 36,000 images of good driving behavior on the jungle track, or around 12,000 images per camera. One difficulty with recording the data was that joysticks and game controllers don't work in the beta simulator as I write this, so I recorded the data using the keyboard, resulting in suboptimal steering angle data: For any given curve, you end up sending a few sharp steering impulses to the car, resulting in some images having steering angles of zero even though the image shows a sharp turn, and some images having too extreme steering angles. The recorded data was still usable enough though.

In order to teach the model to drive on the jungle track, but at the same time not forget how to drive on the lake track, I took a model that was already able to drive well on the lake track (which had been trained for 6 epochs) and trained it for 2 additional epochs on the entire combined training dataset (45,000 images for the lake track plus 36,000 images for the jungle track, minus 10% of that for the validation data). This turned out to be enough to get the model to drive well on both tracks.

Even though it was a problem initially on the jungle track when I didn't limit car's speed, because it would get too fast downhill and miss immediately consecutive turns, surprisingly I managed to get it to a point where the model submitted here can run laps on the jungle track even without any speed limitation if the default throttle is set to 0.2. I still modified the drive.py to ensure a minimum speed of 12 mph (because otherwise watching the car drive on the jungle track is tedious) and a maximum of 24 mph so that the car drives more smoothly on the jungle track. Feel free to set the max speed to above 30 mph though - the driving will be less clean, but it will still work.
