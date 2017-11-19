Vehicle Detection
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labelled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[nonVehicleTrain]: ./nonVehicleTrain.png
[vehicleTrain]: ./vehicleTrain.png
[RGB]: ./RGB.png
[HSV]: ./HSV.png
[HLS]: ./HLS.png
[YUV]: ./YUV.png
[YCrCb]: ./YCrCb.png
[colorHistogram]: ./colorHistogram.png
[colorHistogram2]: ./colorHistogram2.png
[colorHistogram3]: ./colorHistogram3.png

[nonVehicle]: ./nonVehicle.png
[hogNonVehicle]: ./hogNonvehicle.png
[vehicle]: ./vehicle.png
[hogVehicle]: ./hogVehicle.png
[HOGPxPCell2]: ./HOGPxPCell2.png
[HOGPxPCell4]: ./HOGPxPCell4.png
[HOGPxPCell4Errors]: ./HOGPxPCell4Errors.png
[HOGPxPCell8]: ./HOGPxPCell8.png
[HOGPxPCell8Errors]: ./HOGPxPCell8Good.png
[Scale1XSearchWindows]: ./Scale1XSearchWindows.png
[Scale1_5XSearchWindows]: ./Scale1_5XSearchWindows.png
[Scale2XSearchWindows]: ./Scale2XSearchWindows.png
[Heatmap]: ./Heatmap.png
[HeatmapThreshold]: ./HeatThreshold.png
[Final]: ./result.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Running the vehicle detector
The vehicle detection was written as a python program that can be executed on the command-line.  It provides numerous command-line parameters used for,
- Training classifiers
- Trialing different classifier configurations (HOG, Colorspace, Binning)
- Running the vehicle detection algorithm
	- Single image
	- Directory of images
	- Video
- Specifying which classifier(s) to use for the vehicle detection
	- **Note:** A list of classifiers can be specified and used. Their results are combined for the final heatmap and thresholding.
```
$ python VehicleDetection.py -h
usage: VehicleDetection.py [-h] [-t TRAIN_CLASSIFIER] [-e COLORSPACE]
                           [-z TEST_SIZE] [-r ORIENTATION] [-x PX_PER_CELL]
                           [-l CELL_PER_BLOCK] [-g HOG_CHANNEL]
                           [-s SPATIAL_SIZE] [-i HIST_BINS] [-p] [-a] [-b]
                           [-c SINGLE_IMAGE] [-j IMAGE_DIRECTORY]
                           [-v VIDEO_IN] [-o VIDEO_OUT] [-w SEARCH_WINDOW]
                           [-q CLASSIFIERS] [-d]

Vehicle detection

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_CLASSIFIER, --train_classifier TRAIN_CLASSIFIER
                        Train classifier for vehicles (RGB, HSV, LUV, HLS,
                        YUV, YCrCb)
  -n PREV_DETECT --prev_detect PREV_DETECT
  						Set the number of previous detections to include
  -e COLORSPACE, --colorspace COLORSPACE
                        Set the colorspace used when classifying
  -z TEST_SIZE, --test_size TEST_SIZE
                        Set the training test size (as percentage)
  -r ORIENTATION, --orientation ORIENTATION
                        Set HOG orientation
  -x PX_PER_CELL, --px_per_cell PX_PER_CELL
                        Set HOG pixels per cell
  -l CELL_PER_BLOCK, --cell_per_block CELL_PER_BLOCK
                        Set HOG cells per block
  -g HOG_CHANNEL, --hog_channel HOG_CHANNEL
                        Set the HOG Channel (0, 1, 2, ALL)
  -s SPATIAL_SIZE, --spatial_size SPATIAL_SIZE
                        Set the spatial size
  -i HIST_BINS, --hist_bins HIST_BINS
                        Set the histogram bin number
  -p, --spatial_feat    Enable/Disable the use of spatial features
  -a, --hist_feat       Enable/Disable the use of histogram features
  -b, --hog_feat        Enable/Disable the use of HOG features
  -c SINGLE_IMAGE, --single_image SINGLE_IMAGE
                        Vehicle detection on single image
  -j IMAGE_DIRECTORY, --image_directory IMAGE_DIRECTORY
                        Vehicle detection on images stored in directory
  -v VIDEO_IN, --video_in VIDEO_IN
                        Vehicle detection on video
  -o VIDEO_OUT, --video_out VIDEO_OUT
                        Output video file
  -w SEARCH_WINDOW, --search_window SEARCH_WINDOW
                        Window for searching for vehicles
  -q CLASSIFIERS, --classifiers CLASSIFIERS
                        Comma separated list of classifiers to use
  -d, --debug           debug lane finding
```

- Example Training command
    - `python VehicleDetection.py -t trainingData/ -e 'YCrCb' -z 0.2 -r 8 -x 8 -l 2 -g 'ALL' -s 32 -i 32 -p -a -b`

- Example video detection command
    - `python VehicleDetection.py -v <VIDEO_IN> -o <VIDEO_OUT> -q <CLASSIFIER FOLDERS>`

###Writeup / README

### Feature extraction

In order to train and classify images (or sub images) as vehicles, features of the images must be extracted.  
These features are what creates the set of data for determining whether a vehicle is present or not.
A number of different features are used in this project and are discussed here.

####1. Histogram of Oriented Gradients (HOG)
Histogram of oriented gradients calculates the gradients at a finite number of image locations (cells).  Once all orientation of gradients are determined they are finally used to create a histogram of the orientations for the overall image.  

- The code for this step is contained in `VehicleDetection.py` in the `FeatureExtractor` class, line 106. 
	- ```def extractHistOfGradientFeatures(self, image, orientation, pxPerCell, cellPerBlock, visualize, featVector)```
	- It uses the ***skimage.feature*** `hog` function.

- An example of the hog output (showing orientations of an image) is shown below.
![alt text][nonVehicle]
![alt text][hogNonVehicle]
![alt text][vehicle]
![alt text][hogVehicle]

####2. Color histogram
Histograms of individual color channels were calculated as a means to extract features.  This basically produces a map of color channel pixel value frequencies.

- An example of 3 YCrCb channels of an image is shown below.
![alt text][colorHistogram]
![alt text][colorHistogram2]
![alt text][colorHistogram3]

####3. Spatial Data
The feature extracted from this is a flattened (ravel) version of the image (i.e. in a single vector)


### Training
To train the vehicle detector the `VehicleDetection.py` script is run with all training parameters specified.

- An example of training the vehicle detector using the following parameters is shown below.
	- Colorspace: YCrCb
	- Train/Test sample split: 0.2 (20% test)
	- HOG Orientations: 8
	- HOG pixels per cell: 8
	- HOG cells per step: 2
	- HOG color channels to use: ALL - (channel 0, 1, and 2)
	- Spatial size: 32 (converted to a 32x32 image) before `ravel`
	- Histogram bins: 32
	- Use Spatial feature: True
	- Use Color Histogram features: True
	- Use HOG features: True

`python VehicleDetection.py -t trainingData/ -e 'YCrCb' -z 0.2 -r 8 -x 8 -l 2 -g 'ALL' -s 32 -i 32 -p -a -b`

- The training process is as follows:
- Load Vehicle data
	- In total there are 8792 vehicle images.
	- An example is shown here

![alt text][vehicleTrain]

- Load Non-Vehicle data
	- In total there are 8968 non-vehicle images
	- An example is shown here

![alt text][nonVehicleTrain]


Since the vehicle detector can be trained using different input parameters, I was able to test various configurations,

- Different color spaces
- Different spatial sizes
- Different HOG channels
- Different HOG configurations

I found that the RGB, HLS, HSV colorspaces results in a large number of false detections and YUV lacked detection capability.  Therefore, I can assume that these are not optimal in distinguishing the vehicles from non-vehicles.

- RGB
![alt text][RGB]

- HSV
![alt text][HSV]

- HLS
![alt text][HLS]

- YUV
![alt text][YUV]


Using YCrCb showed the greatest benefit in vehicle detections
![alt text][YCrCb]


####2. HOG parameters.

As described above with regard to the classifier parameters, several HOG parameters could also easily be tested with regard to the number of orientations, pixels per cell etc.
In the end I found that using the parameters discussed in the lecture series works sufficiently well.
That is orientation 8, px per cell 8 and steps for cell of 2.

- Different Pixels per Cell
	- Changing this parameters adjusts the resolution of the HOG calculations
	- Using the following values this is what I found
		- 2: Although it looks nice, the training and predictions took significant time and the result were not warranted.
![alt text][HOGPxPCell2]
		- 4: Again, looks nice when viewing the HOG data, however resulted in a lot of false detections.
![alt text][HOGPxPCell4]
![alt text][HOGPxPCell4Errors]
		- 8: This was the default as shown in the lecture series, and produced the best results.
![alt text][HOGPxPCell8]
![alt text][HOGPxPCell8Errors]

I also noted that the best results came from using all HOG channels and not using any particular single channel.

###Sliding Window Search

In order to detect vehicles, a sliding window search was performed.  The image was divided up into N discrete sub-images and the classifier run on each sub-image.  The reason is that the classifier can only detect vehicles in an image of roughly the same size/dimension.  Therefore we must search the large single image as a sum of smaller sub-images.

- See `VehicleDetection.py` for the implementation details.
	- line 362 `def searchWindows(self, image, windows)` 
	- line 308 `def getSlideWindows(self, image, xStartStop=[None, None], yStartStop=[None, None], xyWindow=(64, 64), xyOverlap=(0.5, 0.5))` 

Below is a depiction of the sub-windows that are used in the detection pipeline.

 - 1X scaling
 	- detect vehicles (roughly 64x64 pixels) in the native image resolution
 	- Note that since these vehicles will be small, we can restrict the search to a sub-section of the image that represents a location where vehicles would be further away
 ![alt text][Scale1XSearchWindows]
 - 1.5X scaling
 	- detect vehicles at 1.5x the scaling.  This will detect larger vehicles (i.e. closer).
 	- Again we can restrict the search to a vehicles medium distance away - therefore 2/3s of the image
 ![alt text][Scale1_5XSearchWindows]
 - 2x scaling
    - Detect the largest of the vehicles (closest)
    - Since there are not so many search windows here, I choose to just search the entire search area and not restrict the search to a sub-section
 ![alt text][Scale2XSearchWindows]



### Pipeline
My detection pipleline in a brief summary is as follows.

- Create search windows.  I use 3 different scales for search windows (described above)
	- 1x search
	- 1.5 x search
	- 2 x search
- At each window the classifier predicts the detection
	- I use a combination of 2 classifiers for detection and average the result.  Therefore, for a detection to be presented, it must be detected in both classifiers.  The pipeline allows one to specify any number of classifiers whose results will be summed and evaluted.
	- The classifiers I utilise are.
		- Classifier 1:
			- Colorspace: YCrCb
			- Train/Test sample split: 0.2 (20% test)
			- HOG Orientations: 8
			- HOG pixels per cell: 8
			- HOG cells per step: 2
			- HOG color channels to use: ALL - (channel 0, 1, and 2)
			- Spatial size: 32 (converted to a 32x32 image) before `ravel`
			- Histogram bins: 32
			- Use Spatial feature: True
			- Use Color Histogram features: True
			- Use HOG features: True
		- Classifier 2:
			- Colorspace: YCrCb
			- Train/Test sample split: 0.2 (20% test)
			- HOG Orientations: 16
			- HOG pixels per cell: 8
			- HOG cells per step: 2
			- HOG color channels to use: ALL - (channel 0, 1, and 2)
			- Spatial size: 64 (converted to a 64x64 image) before `ravel`
			- Histogram bins: 64
			- Use Spatial feature: True
			- Use Color Histogram features: True
			- Use HOG features: True
- At each image, the detections from the last `N` image frames are also included in the detection so that the vehicle needs to be,
	- Detected by multiple classifiers
	- Detected by the last `N` frames
	- In my case, I have tested this for 2, 3 and 4 previous frame detections.
		- Anything over 2 seems to give reasonable results.
- From detections, create heatmap
	- for each detection window add 1 to the pixel value
![alt text][Heatmap]
- Threshold the heatmap
	- Only include heat signature that is greater than a set amount (i.e. has been detected in mutliple overlapping search windows)
![alt text][HeatmapThreshold]
- Create labels from heatmap
	- Determines bounding boxes around "hot" areas in the image.
	- This becomes the vehicle detection bounding boxes
- Finally overlay detections on input image.
![alt text][Final]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
- Here's a [link to my debug video result](./output_project_debug.mp4)
- Here's a [link to my video result](./project_output_final.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Filtering of false positives is done in a number of ways.
- Producing a heatmap (as described earlier) and then thresholding this heatmap such that a minimum number of search windows detections are required before a vehicle detection is confirmed.
- Combining the last `N` detection frames, such that a detection must persist across multiple images before it is considered a "true" vehicle detection.
- Combining the detections from multiple classifiers, such that each classifier must detect the same object(vehicle) before it is considered a "true" vehicle detection.

Combining these 3 techniques has produced a reasonable robust solution.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The issues I faced were
 - False detections
 	- Using a single classifier, with no filtering shows a large number of false detections.
 	- Therefore, it was required to add some output filtering (heatmap thresholding), and also combining results from multiple classifiers, and previous detections.
 - Missed detections
 	- Initially, there were many instances, were vehicles were not detected
 		- The white car was often not detected
 			- To solve this I needed to change the colorspace used (I ended up used the YCrCb colorspace)
 		- Random non-detections
 			- To solve this I included detections on multiple image scales, mutliple classifieres, across multiple image frames

 Likely issues
 	- Detecting vehicles that have not been trained on
 		- For instance, Buses, Trucks, Trains, Caravans, Motorhomes, etc, etc
 		- I would guess that since these have not been seen in the training data and their shapes, colors etc are vastly different, this could cause issues
 	- Detecting bicycles, motorbikes
 		- Same as above, however these are also much smaller, so the resolution of the HOG may become an issue also
 	- Different weather conditions
 		- Raining, Snow, Hail.  These would all make the vehicles look different (shiny, covered in snow etc)
 		- I would assume that this could also cause contrast issues with the background and the road surface.
 	- Vehicles travelling toward the vehicle.
 		- This project only considered vehicles traveling away from the vehicle/travelling in the same direction
 		- Vehicles travelling toward the vehicle would have different profile and different color profile (due to windscreen) than vehicle traveling in the same direction.
 	- Real-time detection
 		- At the moment, the pipeline that I use is not efficient and is not anywhere close to real-time.
 		- Lots of improvements would need to be made to have this usable in a real-world situation


 Making more robust
    - I'd like to explore the idea of combining more classifiers/neural networks and have a combination system that determines if a detection is "true" or not.
    - More training data convering more situations and vehicle types.

