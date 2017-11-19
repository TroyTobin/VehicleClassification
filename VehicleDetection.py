import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import time
import pickle
from glob import glob
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
import argparse
from moviepy.editor import VideoFileClip
from time import sleep

CLASSIFIER_LABEL        = "Classifier"
X_SCALER_LABEL          = "XScaler"
COLORSPACE_LABEL        = "Colorspace"
ORIENT_LABEL            = "Orientation"
PIXELS_PER_CELL_LABEL   = "PixelsPerCell"
CELL_PER_BLOCK_LABEL    = "cellsPerBlock"
SPATIAL_SIZE_LABEL      = "SpatialSize"
HISTOGRAM_BINS_LABEL    = "HistogramBins"
HOG_CHANNEL_LABEL       = "HOGChannel"
SPATIAL_FEAT_LABEL      = "SpatialFeatures"
HISTOGRAM_FEAT_LABEL    = "HistogramFeatures"
HOG_FEAT_LABEL          = "HOGFeatures"

class Logger():
    '''
    Class used for logging and debugging
    '''
    def __init__(self, debug, outputDir):
        '''
        Initialize the logger class

        '''
        self.debug = debug
        self.outputDir = outputDir

        # Check that the output directory exists
        # If not, create it
        if (os.path.exists(self.outputDir) == False):
            try:
                os.mkdir(self.outputDir)
                self.Print("Created output directory (%s)\n" % (self.outputDir))
            except Exception as ex:
                self.Print("Failed creating output directory (%s)\n" % (self.outputDir))


    def Print(self, msg):
        '''
        Debug logging
        '''
        if (self.debug == True):
            print (msg)

    def printStart(self, msg):
        '''
        Special message for start of function
        '''
        self.Print("START -> %s\n" % (msg))

    def printEnd(self, msg):
        '''
        Special message for start of function
        '''
        self.Print("END -> %s\n" % (msg))

    def outputImage(self, image, fileName):
        '''
        Save image to disk
        '''
        if (self.debug == True):
            # Open file to write to
            try:
                fHandle = open("%s/%s" % (self.outputDir, fileName), 'w')
                fHandle.write(image)
                fHandle.close()
            except (IOError, OSError) as ex:
                print("Failed to output image file (%s)\n" % (self.outputDir))

    def outputFigure(self, figure, fileName):
        '''
        Save plot figure to disk
        '''        
        if (self.debug == True):
            figure.savefig(fileName, dpi=figure.dpi)

class FeatureExtractor():
    '''
    Class for extracting features from images
    '''
    def __init__(self, debug):
        '''
        Initialize the feature extractor.
        Only need the logger at this stage.
        '''
        self.logger = Logger(debug, "FeatureExtraction")

    def extractHistOfGradientFeatures(self, image, orientation, pxPerCell, cellPerBlock, visualize, featVector):
        '''
        Extract the histogram of gradients of the image.
        This uses the skimage module's `hog` function.

        Will return a tuple (features, visImage) if self.visualize is True
        '''
        self.logger.printStart("extractHistOfGradientFeatures (%s, %d, %d, %s)" % (orientation, pxPerCell, cellPerBlock, visualize))
        result = hog(image, orientations=orientation,
                            pixels_per_cell=(pxPerCell, pxPerCell),
                            cells_per_block=(cellPerBlock, cellPerBlock),
                            transform_sqrt=True,
                            visualise=visualize,
                            feature_vector=featVector)
 
        # If visualizing the result, do so here.
        if (visualize == True) and (len(result) == 2):
            plt.imshow(result[1], cmap='gray')
            plt.show()
            result = result[0]


        self.logger.printEnd("extractHistOfGradientFeatures")
        return result
        

    def extractBinnedSpatialData(self, image, spatialSize=(32, 32)):
        '''
        Reduce dimensions of the image (binning) and return 1-dimension array of the data.
        '''
        self.logger.printStart("extractBinnedSpatialData (%s, %s)" % (image.shape, spatialSize))
        #self.logger.outputImage(image, "extractBinnedSpatialData_Input.png")

        result = cv2.resize(image, spatialSize).ravel()
        
        #figure = plt.figure()
        #plt.plot(result)
        #self.logger.outputFigure(figure, "extractBinnedSpatialData_Features.png")

        self.logger.printEnd("extractBinnedSpatialData")
        return result


    def extractColorHistogramData(self, image, binsNum=32):
        '''
        Extract the histogram data for the color channels.
        '''
        self.logger.printStart("extractColorHistogramData (%d)" % (binsNum))
        #self.logger.outputImage(image, "extractColorHistogramData_Input.png")
        binsRange=(-1, 1)
        channel1Histogram = np.histogram(image[:,:,0], bins=binsNum, range=binsRange)
        channel2Histogram = np.histogram(image[:,:,1], bins=binsNum, range=binsRange)
        channel3Histogram = np.histogram(image[:,:,2], bins=binsNum, range=binsRange)

        # Concatenate the histograms into a single feature vector
        result = np.concatenate((channel1Histogram[0], channel2Histogram[0], channel3Histogram[0]))
        
        # Return the individual histograms, bin_centers and feature vector
        self.logger.printEnd("extractColorHistogramData")
        return result

    def extractImageFeatures(self, image, hogInputFeatures, colorSpace='RGB', 
                                   spatialFeatures=True, spatialSize=(32, 32), 
                                   histogramFeatures=True, binsNum=32, 
                                   hogFeatures=True, orientation=9, pxPerCell=8, cellPerBlock=2, hogColorChannel='ALL'):
        '''
        Extract features for set of images
        '''
        self.logger.printStart("extractImageFeatures (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (colorSpace, spatialFeatures, spatialSize, histogramFeatures, binsNum, hogFeatures, orientation, pxPerCell, cellPerBlock, hogColorChannel))
       
        allFeatures = []
        imageFeatures = []
        featureImage = np.copy(image)
        # Extract spacial features
        if (spatialFeatures == True):
            spatialFeatures = self.extractBinnedSpatialData(featureImage, spatialSize=spatialSize)
            imageFeatures.append(spatialFeatures)

        # Extract histogram of colors features
        if (histogramFeatures == True):
            histogramFeatures = self.extractColorHistogramData(featureImage, binsNum=binsNum)
            imageFeatures.append(histogramFeatures)

        # Extract histogram of gradients features
        if (hogFeatures == True):
            if (hogColorChannel == 'ALL'):
                hogOutputFeatures = []
                for channel in range(featureImage.shape[2]):
                    hogOutputFeatures.append(hogInputFeatures[channel])

                hogOutputFeatures = np.ravel(hogOutputFeatures)
            else:
                hogOutputFeatures = hogInputFeatures[hogColorChannel]
                hogOutputFeatures = np.ravel(hogOutputFeatures)

            # Append the new feature vector to the features list
            imageFeatures.append(hogOutputFeatures)

        # Append all extracted features
        allFeatures.append(np.concatenate(imageFeatures))

        self.logger.printEnd("extractImageFeatures")

        # Return list of feature vectors
        return allFeatures


    def extractFeatures(self, imageFileNames, colorspace='RGB', 
                              spatialFeatures=True, spatialSize=(32, 32), 
                              histogramFeatures=True, binsNum=32, 
                              hogFeatures=True, orientation=9, pxPerCell=8, cellPerBlock=2, hogColorChannel='ALL'):
        '''
        Extract features for set of images
        '''
        self.logger.printStart("extractFeatures (%d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (len(imageFileNames), colorspace, spatialFeatures, spatialSize, histogramFeatures, binsNum, hogFeatures, orientation, pxPerCell, cellPerBlock, hogColorChannel))
        allFeatures = []

        # Iterate through the list of images
        for imageFileName in imageFileNames:
            imageFeatures = []

            # Read in each image one by one
            image = mpimg.imread(imageFileName)
            test_image = np.copy(image)

            imageFeatures = []
            # Apply color conversion if other than 'RGB'
            if (colorspace != 'RGB'):
                if (colorspace == 'HSV'):
                    featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
                elif (colorspace == 'LUV'):
                    featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2LUV)
                elif (colorspace == 'HLS'):
                    featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2HLS)
                elif (colorspace == 'YUV'):
                    featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2YUV)
                elif (colorspace == 'YCrCb'):
                    featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2YCR_CB)
                else:
                    self.logger.Print("extractImageFeatures: Color-space not supported (%s)\n" % (colorSpace))
            else:
                featureImage = np.copy(test_image)

             # calculate hog once-off
            ch1 = featureImage[:,:,0]
            ch2 = featureImage[:,:,1]
            ch3 = featureImage[:,:,2]
            hogOutputCh1 = self.extractHistOfGradientFeatures(ch1, orientation=orientation, pxPerCell=pxPerCell, cellPerBlock=cellPerBlock, visualize=False, featVector=False)
            hogOutputCh2 = self.extractHistOfGradientFeatures(ch2, orientation=orientation, pxPerCell=pxPerCell, cellPerBlock=cellPerBlock, visualize=False, featVector=False)
            hogOutputCh3 = self.extractHistOfGradientFeatures(ch3, orientation=orientation, pxPerCell=pxPerCell, cellPerBlock=cellPerBlock, visualize=False, featVector=False)

            imageFeatures += self.extractImageFeatures(featureImage, [hogOutputCh1, hogOutputCh2, hogOutputCh3], colorspace, spatialFeatures, spatialSize, histogramFeatures, binsNum, hogFeatures, orientation, pxPerCell, cellPerBlock, hogColorChannel)
            self.logger.Print("extractFeatures Image features (%s)" % (imageFeatures))

            # Append all extracted features
            allFeatures.append(np.concatenate(imageFeatures))

        self.logger.printEnd("extractFeatures")

        # Return list of feature vectors
        return allFeatures


class Classifier():
    '''
    Class for training the vehicle detection classifier
    '''
    def __init__(self, calFile, debug):
        '''
        Initialize the vehicle classifier
        '''
        self.calibrationFile  = calFile
        self.featureExtractor = FeatureExtractor(debug)
        self.logger           = Logger(debug, "Classifier")
        self.classifier       = None
        self.XScaler          = None
        self.colorSpace       = None
        self.orientation      = None
        self.pxPerCell        = None
        self.cellPerBlock     = None
        self.spatialSize      = None
        self.histBins         = None
        self.hogChannel       = None
        self.spatialFeat      = None
        self.histogramFeat    = None
        self.hogFeat          = None

    def getSlideWindows(self, image, xStartStop=[None, None], yStartStop=[None, None], 
                              xyWindow=(64, 64), xyOverlap=(0.5, 0.5)):
        '''
        Get a list of search windows
        '''
        self.logger.printStart("getSlideWindows (%s, %s, %s, %s)\n" % (xStartStop, yStartStop, xyWindow, xyOverlap))

        # If x and/or y start/stop positions not defined, set to image size
        if xStartStop[0] == None:
            xStartStop[0] = 0
        if xStartStop[1] == None:
            xStartStop[1] = image.shape[1]
        if yStartStop[0] == None:
            yStartStop[0] = 0
        if yStartStop[1] == None:
            yStartStop[1] = image.shape[0]
        
        # Compute the span of the region to be searched    
        xSpan = xStartStop[1] - xStartStop[0]
        ySpan = yStartStop[1] - yStartStop[0]
        
        # Compute the number of pixels per step in x/y
        nxPxPerStep = np.int(xyWindow[0]*(1 - xyOverlap[0]))
        nyPxPerStep = np.int(xyWindow[1]*(1 - xyOverlap[1]))

        # Compute the number of windows in x/y
        nxBuffer = np.int(xyWindow[0]*(xyOverlap[0]))
        nyBuffer = np.int(xyWindow[1]*(xyOverlap[1]))

        nxWindows = np.int((xSpan - nxBuffer)/nxPxPerStep)
        nyWindows = np.int((ySpan - nyBuffer)/nyPxPerStep)

        # Initialize a list to append window positions to
        windowList = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(nyWindows):
            for xs in range(nxWindows):
                # Calculate window position
                startx = xs*nxPxPerStep + xStartStop[0]
                endx = startx + xyWindow[0]
                starty = ys*nyPxPerStep + yStartStop[0]
                endy = starty + xyWindow[1]
                
                # Append window position to list
                windowList.append(((startx, starty), (endx, endy)))

        self.logger.printEnd("getSlideWindows")

        # Return the list of windows
        return windowList

    def searchWindows(self, image, windows):
        '''
        Search each window for vehicles.
        '''
        self.logger.printStart("searchWindows")

        #1) Create an empty list to receive positive detection windows
        onWindows = []

        # Check that the classifier has been trained
        if (self.classifier is None):
            self.logger.Print("searchWindows: Classifier is not yet trained")
            return 

        
        test_image = np.copy(image)

        # Apply color conversion if other than 'RGB'
        if (self.colorSpace != 'RGB'):
            if (self.colorSpace == 'HSV'):
                featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
            elif (self.colorSpace == 'LUV'):
                featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2LUV)
            elif (self.colorSpace == 'HLS'):
                featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2HLS)
            elif (self.colorSpace == 'YUV'):
                featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2YUV)
            elif (self.colorSpace == 'YCrCb'):
                featureImage = cv2.cvtColor(test_image, cv2.COLOR_RGB2YCR_CB)
            else:
                self.logger.Print("extractImageFeatures: Color-space not supported (%s)\n" % (colorSpace))
        else:
            featureImage = np.copy(test_image)

        # Calculate HOG once-off for the entire image and just sub-sample when needed below.
        ch1 = featureImage[:,:,0]
        ch2 = featureImage[:,:,1]
        ch3 = featureImage[:,:,2]
        hog1 = self.featureExtractor.extractHistOfGradientFeatures(ch1, self.orientation, self.pxPerCell, self.cellPerBlock, visualize=False, featVector=False)
        hog2 = self.featureExtractor.extractHistOfGradientFeatures(ch2, self.orientation, self.pxPerCell, self.cellPerBlock, visualize=False, featVector=False)
        hog3 = self.featureExtractor.extractHistOfGradientFeatures(ch3, self.orientation, self.pxPerCell, self.cellPerBlock, visualize=False, featVector=False)
        
        self.featureExtractor.extractImageFeatures(featureImage, [hog1, hog2, hog2], colorSpace=self.colorSpace, 
                                                                                   spatialFeatures=self.spatialFeat, spatialSize=(self.spatialSize, self.spatialSize), 
                                                                                   histogramFeatures=self.histogramFeat, binsNum=self.histBins,
                                                                                   hogFeatures=self.hogFeat, orientation=self.orientation, pxPerCell=self.pxPerCell, cellPerBlock=self.cellPerBlock, hogColorChannel=self.hogChannel)
        #2) Iterate over all windows in the list
        for window in windows:
            startTime = time.time()
            #3) Extract the test window from original image
            testImage = cv2.resize(featureImage[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

            # convert froom pixels to HOG windows
            cellXStart = math.floor(window[0][0]/self.pxPerCell)
            cellXEnd   = math.floor(window[1][0]/self.pxPerCell) - 1
            cellYStart = math.floor(window[0][1]/self.pxPerCell)
            cellYEnd   = math.floor(window[1][1]/self.pxPerCell) - 1
            
            hog_feat1 = hog1[cellYStart:cellYEnd, cellXStart:cellXEnd].ravel() 
            hog_feat2 = hog2[cellYStart:cellYEnd, cellXStart:cellXEnd].ravel() 
            hog_feat3 = hog3[cellYStart:cellYEnd, cellXStart:cellXEnd].ravel()

            #4) Extract features for that window using single_img_features()
            imageFeatures = []
            imageFeatures += self.featureExtractor.extractImageFeatures(testImage, [hog_feat1, hog_feat2, hog_feat3], colorSpace=self.colorSpace, 
                                                                                   spatialFeatures=self.spatialFeat, spatialSize=(self.spatialSize, self.spatialSize), 
                                                                                   histogramFeatures=self.histogramFeat, binsNum=self.histBins,
                                                                                   hogFeatures=self.hogFeat, orientation=self.orientation, pxPerCell=self.pxPerCell, cellPerBlock=self.cellPerBlock, hogColorChannel=self.hogChannel)
            
            #5) Scale extracted features to be fed to classifier
            testFeatures = self.XScaler.transform(np.array(imageFeatures).reshape(1, -1))

            #6) Predict using your classifier
            prediction = self.classifier.predict(testFeatures)

            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                onWindows.append(window)

            endTime = time.time()
            #print("searchTime", (endTime - startTime))

        self.logger.printEnd("searchWindows")

        #8) Return windows for positive detections
        return onWindows
       
    @classmethod 
    def generateHeatmap(cls, image, boundingBoxes):
        '''
        Create headmap from all vehicle detections
        '''

        heatmap = np.zeros_like(image)
        for boundingBox in boundingBoxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[boundingBox[0][1]:boundingBox[1][1], boundingBox[0][0]:boundingBox[1][0]] += 1

        # Return updated heatmap
        return heatmap
    
    @classmethod
    def applyHeatmapThreshold(cls, heatmap, threshold):
        '''
        Apply a threashold to a heatmap.
        If threshold not met, set pixel to 0 (empty)
        '''

        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0

        # Return thresholded map
        return heatmap

    @classmethod 
    def overlayBoxes(cls, image, boundingBoxes, color=(0, 0, 255), thickness=6):
        '''
        Overlay bounding boxes in the input image
        '''
        # Make a copy of the image
        result = np.copy(image)

        # Iterate through the bounding boxes
        for bbox in boundingBoxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(result, bbox[0], bbox[1], color, thickness)

        # Return the image copy with boxes drawn
        return result

    @classmethod
    def overlayLabeledBboxes(cls, img, labels):
        '''
        Overlay labled, vehicle detections boxes
        '''

        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):

            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

        # Return the image
        return img

    def load(self):
        '''
        Load a pre-trained classifier
        '''
        distPickle        = pickle.load(open(self.calibrationFile, "rb" ))

        self.classifier    = distPickle[CLASSIFIER_LABEL]
        self.XScaler       = distPickle[X_SCALER_LABEL]
        self.colorSpace    = distPickle[COLORSPACE_LABEL]
        self.orientation   = distPickle[ORIENT_LABEL]
        self.pxPerCell     = distPickle[PIXELS_PER_CELL_LABEL]
        self.cellPerBlock  = distPickle[CELL_PER_BLOCK_LABEL]
        self.spatialSize   = distPickle[SPATIAL_SIZE_LABEL]
        self.histBins      = distPickle[HISTOGRAM_BINS_LABEL]
        self.hogChannel    = distPickle[HOG_CHANNEL_LABEL]
        self.spatialFeat   = distPickle[SPATIAL_FEAT_LABEL]
        self.histogramFeat = distPickle[HISTOGRAM_FEAT_LABEL]
        self.hogFeat       = distPickle[HOG_FEAT_LABEL]


    def train(self, trainingDirectory, testSize, orientation, pxPerCell, cellPerBlock, hogChannel, spatialSize, histBins, spatialFeat, histFeat, hogFeat, colorspace='LUV'):
        '''
        Traing the vehicle detection classifier
        Can be RGB, HSV, LUV, HLS, YUV, YCrCb, or mulitple of these
        '''
        self.logger.printStart("train (%s)" % (trainingDirectory))

        # Read in cars and notcars
        vehicleDirectory    = "%s/vehicles/" % (trainingDirectory)
        nonVehicleDirectory = "%s/nonVehicles/" % (trainingDirectory)

        vehicleImageFiles = []
        # Iterate through the sub-directories looking for training images
        for subdir in os.listdir(vehicleDirectory):
            vehicleImageFiles = np.concatenate((vehicleImageFiles, glob("%s/%s/*.jpeg" % (vehicleDirectory, subdir))))
            vehicleImageFiles = np.concatenate((vehicleImageFiles, glob("%s/%s/*.png" % (vehicleDirectory, subdir))))


        nonVehicleImageFiles = []
        # Iterate through the sub-directories looking for training images
        for subdir in os.listdir(nonVehicleDirectory):
            nonVehicleImageFiles = np.concatenate((nonVehicleImageFiles, glob("%s/%s/*.jpeg" % (nonVehicleDirectory, subdir))))
            nonVehicleImageFiles = np.concatenate((nonVehicleImageFiles, glob("%s/%s/*.png" % (nonVehicleDirectory, subdir))))

        # Extract the features of th vehicle and non-vehicle samples.
        vehicleFeatures     = self.featureExtractor.extractFeatures(vehicleImageFiles, colorspace=colorspace, 
                                                                                       spatialFeatures=spatialFeat, spatialSize=(spatialSize, spatialSize), 
                                                                                       histogramFeatures=histFeat, binsNum=histBins,
                                                                                       hogFeatures=hogFeat, orientation=orientation, pxPerCell=pxPerCell, cellPerBlock=cellPerBlock, hogColorChannel=hogChannel)

        nonvehicleFeatures  = self.featureExtractor.extractFeatures(nonVehicleImageFiles, colorspace=colorspace, 
                                                                                          spatialFeatures=spatialFeat, spatialSize=(spatialSize, spatialSize), 
                                                                                          histogramFeatures=histFeat, binsNum=histBins,
                                                                                          hogFeatures=hogFeat, orientation=orientation, pxPerCell=pxPerCell, cellPerBlock=cellPerBlock, hogColorChannel=hogChannel)

        X = np.vstack((vehicleFeatures, nonvehicleFeatures)).astype(np.float64)
        
        # Fit a per-column scaler
        XScaler = StandardScaler().fit(X)
        
        # Apply the scaler to X
        scaledX = XScaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(vehicleFeatures)), np.zeros(len(nonvehicleFeatures))))

        # Split up data into randomized training and test sets
        randomState = np.random.randint(0, 100)
        XTrain, XTest, YTrain, YTest = train_test_split(scaledX, y, test_size=testSize, random_state=randomState)

        self.logger.Print("train: (XTrain, XTest, YTrain, YTest)  (%d, %d, %d, %d)" % (len(XTrain), len(XTest), len(YTrain), len(YTest)))
        self.logger.Print("train: Feature vector length (%d)" % (len(XTrain[0])))

        # Use a linear SVC 
        svc = LinearSVC()
        
        # Check the training time for the SVC
        startTrainTime = time.time()

        svc.fit(XTrain, YTrain)

        endTrainTime = time.time()

        self.logger.Print("train: Feature vector length (%d)" % (len(XTrain[0])))
        self.logger.Print("train: %d Seconds to train SVC" % (round(endTrainTime - startTrainTime, 2)))
        self.logger.Print("train: Accuracy of SVC (%f)" % (round(svc.score(XTest, YTest), 4)))

        # Set the internal parameters of the classifier
        self.classifier    = svc
        self.XScaler       = XScaler
        self.colorSpace    = colorspace
        self.orientation   = orientation
        self.pxPerCell     = pxPerCell
        self.cellPerBlock  = cellPerBlock
        self.spatialSize   = spatialSize
        self.histBins      = histBins
        self.hogChannel    = hogChannel
        self.spatialFeat   = spatialFeat
        self.histogramFeat = histFeat
        self.hogFeat       = hogFeat

        # Save the classifier to disk for use at a later time
        distPickle = {}
        distPickle[CLASSIFIER_LABEL]      = svc
        distPickle[X_SCALER_LABEL]        = XScaler
        distPickle[COLORSPACE_LABEL]      = colorspace
        distPickle[ORIENT_LABEL]          = orientation
        distPickle[PIXELS_PER_CELL_LABEL] = pxPerCell
        distPickle[CELL_PER_BLOCK_LABEL]  = cellPerBlock
        distPickle[SPATIAL_SIZE_LABEL]    = spatialSize
        distPickle[HISTOGRAM_BINS_LABEL]  = histBins
        distPickle[HOG_CHANNEL_LABEL]     = hogChannel
        distPickle[SPATIAL_FEAT_LABEL]    = spatialFeat
        distPickle[HISTOGRAM_FEAT_LABEL]  = histFeat
        distPickle[HOG_FEAT_LABEL]        = hogFeat

        dirName = 'classifier/'

        # The save location will be a specific path based on the train characteristics
        newDirname = "%s_%d_%s_%s_%s_%s" % (colorspace, spatialSize, spatialFeat, histFeat, hogFeat, dirName)
        fName = "%s%s" % (newDirname, self.calibrationFile)
        print(fName)

        # Create the classifier directory if it does not yet exist.
        if (os.path.exists(newDirname) == False):
            try:
                os.mkdir(newDirname)
                print("Created output directory (%s)\n" % (newDirname))
            except Exception as ex:
                print("Failed creating output directory (%s)\n" % (newDirname))

        pickle.dump(distPickle, open(fName, "wb" ))

        self.logger.printEnd("train")


    def testClassifier(self, testDirectory, scaler):
        '''
        Test the classifier some test images
        '''
        print("testClassifier (%s)" % (testDirectory))
        self.logger.printEnd("testClassifier (%s)" % (testDirectory))
        
        # Load the test images
        testImages = glob("%s/*.jpg" % (testDirectory))

        # For each image, load it and "find the cars"       
        for testImage in testImages:
            image = mpimg.imread(testImage)
            outputImage = np.copy(image)

            # Detect any windows containing vehicles
            detectionBoxes = self.find_cars(image, 400, 656, scaler)

            # If vehicles found, overlay boxes and show the user.
            if (detectionBoxes is not None):
                windowImg = self.overlayBoxes(outputImage, detectionBoxes, color=(0, 0, 255), thickness=6)                    
                plt.imshow(windowImg)
                plt.show()

        self.logger.printEnd("testClassifier")

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, searchWindows=None):
        draw_img = np.copy(img)

        # Only calculate the search window if not specified.
        if (searchWindows is None):
            searchWindows = []
            searchWindows += self.getSlideWindows(img, xStartStop=[None, None], yStartStop=[ystart, ystop], xyWindow=(64, 64), xyOverlap=(0.75, 0.75))
                
        
        hotWindows     = self.searchWindows(img, searchWindows)  
        return hotWindows, searchWindows


def detectVehicles(image, standalone=False):

    startTime = time.time()
    ystart = 400
    ystop = 656
    scales = [1, 1.5, 2]
    if (np.max(image[0]) > 1):
       _image = image.astype(np.float32)/255

    #plt.imshow(image)
    #plt.show()
    if (detectVehicles.vehicleClassifier is not None):
        # scan the image and find cars
        heat = np.zeros_like(_image)
        detectionBoxes = []
        for classifier in detectVehicles.vehicleClassifier:
            for scale in scales:
                imshape = _image.shape
                
                #resizeStartTime = time.time()
                tosearch = cv2.resize(_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

                searchWindowKey = "sw_%s" % (scale)
                searchWindows = None
                if (searchWindowKey in detectVehicles.searchWindows):
                    searchWindows = detectVehicles.searchWindows[searchWindowKey] 

                _ystop = ystop - 2*(float((ystop - ystart)*(2 - scale))/3)

                _detectionBoxes, searchWindows = classifier.find_cars(tosearch, np.int(ystart/scale), np.int(_ystop/scale), searchWindows)

                if (searchWindowKey not in detectVehicles.searchWindows):
                    detectVehicles.searchWindows[searchWindowKey] = searchWindows
                
                d = (np.array(_detectionBoxes)*scale).astype(int)
                _detectionBoxes = [(tuple(x[0]), tuple(x[1])) for x in d]
                detectionBoxes += _detectionBoxes


        _detectionBoxes = [x for x in detectionBoxes]
        # inlcude the last N detections in this one 
        # (i.e. the vehicle needs to be detected in at least a few of the last frames to be valid)
        for prevDetectionBoxes in detectVehicles.prevDetectionBoxes:
            detectionBoxes += prevDetectionBoxes
        detectVehicles.prevDetectionBoxes.append(_detectionBoxes)

        # make sure we are only using the most recent detections
        while len(detectVehicles.prevDetectionBoxes) > detectVehicles.prevDetections:
            detectVehicles.prevDetectionBoxes.pop(0)

        # Add heat to each box in box list
        heat += Classifier.generateHeatmap(image, detectionBoxes)

        # Apply threshold to help remove false positives
        if (standalone is True):
            heatThreshold = Classifier.applyHeatmapThreshold(heat, 2)
        else:
            heatThreshold = Classifier.applyHeatmapThreshold(heat, 3*(len(detectVehicles.vehicleClassifier) + detectVehicles.prevDetections - 2))

        # Visualize the heatmap when displaying
        heatmap = np.clip(heatThreshold, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = Classifier.overlayLabeledBboxes(np.copy(image), labels)

        if (standalone is True):
            return draw_img, image, detectionBoxes, heat, heatThreshold
        else:
            return draw_img
    else:
        filename = "testImages/Test_%04d.jpg" % (detectVehicles.count)
        detectVehicles.count += 1
        mpimg.imsave(filename, image)
        return image
detectVehicles.vehicleClassifier = None
detectVehicles.prevDetectionBoxes = []
detectVehicles.searchWindows = {}
detectVehicles.count = 1
detectVehicles.prevDetections = 3


# Main processing
# Parse commandline arguments and calibrate the lane detection accordingly  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle detection')
    parser.add_argument('-t', '--train_classifier', help='Train classifier for vehicles (RGB, HSV, LUV, HLS, YUV, YCrCb)')
    parser.add_argument('-n', '--prev_detect',      help='Set the number of previous detections to include')    
    parser.add_argument('-e', '--colorspace',       help='Set the colorspace used when classifying')
    parser.add_argument('-z', '--test_size',        help='Set the training test size (as percentage)')
    parser.add_argument('-r', '--orientation',      help='Set HOG orientation')
    parser.add_argument('-x', '--px_per_cell',      help='Set HOG pixels per cell')
    parser.add_argument('-l', '--cell_per_block',   help='Set HOG cells per block')
    parser.add_argument('-g', '--hog_channel',      help='Set the HOG Channel (0, 1, 2, ALL)')
    parser.add_argument('-s', '--spatial_size',     help='Set the spatial size')
    parser.add_argument('-i', '--hist_bins',        help='Set the histogram bin number')
    parser.add_argument('-p', '--spatial_feat',     action="store_true", help='Enable/Disable the use of spatial features')
    parser.add_argument('-a', '--hist_feat',        action="store_true", help='Enable/Disable the use of histogram features')
    parser.add_argument('-b', '--hog_feat',         action="store_true", help='Enable/Disable the use of HOG features')
    parser.add_argument('-c', '--single_image',     help='Vehicle detection on single image')
    parser.add_argument('-j', '--image_directory',  help='Vehicle detection on images stored in directory')
    parser.add_argument('-v', '--video_in',         help='Vehicle detection on video')
    parser.add_argument('-o', '--video_out',        help='Output video file')
    parser.add_argument('-w', '--search_window',    help='Window for searching for vehicles')
    parser.add_argument('-q', '--classifiers',      help='Comma separated list of classifiers to use')
    parser.add_argument('-d', '--debug',            action="store_true", help='debug lane finding')

    args = parser.parse_args()
        
    CLASSIFIER_FILE         = 'classifier_pickle.p'

    if (args.prev_detect is not None):
        detectVehicles.prevDetections = int(args.prev_detect)

    ## Create/Load the classifier
    if (args.train_classifier is not None):
        # Create the classifier
        testSize     = args.test_size
        orientation  = args.orientation
        pxPerCell    = args.px_per_cell
        cellPerBlock = args.cell_per_block
        hogChannel   = args.hog_channel
        spatialSize  = args.spatial_size
        histBins     = args.hist_bins
        spatialFeat  = args.spatial_feat
        histFeat     = args.hist_feat
        hogFeat      = args.hog_feat
        colorspace   = args.colorspace

        # Sanity check that the input parameters have been provided
        if ((testSize     is None) or 
            (orientation  is None) or 
            (pxPerCell    is None) or 
            (cellPerBlock is None) or 
            (hogChannel   is None) or 
            (spatialSize  is None) or 
            (histBins     is None) or 
            (spatialFeat  is None) or 
            (histFeat     is None) or 
            (hogFeat      is None) or
            (colorspace   is None)):
            print ("Need to provide the following to train the vehicle classifier (test_size, orientation, px_per_cell, cell_per_block, hog_channel, spatial_size, hit_bins, spatial_feat, hist_feat, hog_feat)")
            parser.print_help()
            sys.exit(1)

        testSize     = float(testSize)
        orientation  = int(orientation)
        pxPerCell    = int(pxPerCell)
        cellPerBlock = int(cellPerBlock)
        if (hogChannel != "ALL"):
            hogChannel   = int(hogChannel)
        spatialSize  = int(spatialSize)
        histBins     = int(histBins)

        print ("Training classifier with parameters:")
        print ("colorspace:", colorspace)
        print ("testSize:", testSize)
        print ("orientation:", orientation)
        print ("pxPerCell:", pxPerCell)
        print ("cellPerBlock:", cellPerBlock)
        print ("hogChannel:", hogChannel)
        print ("spatialSize:", spatialSize)
        print ("histBins:", histBins)
        print ("spatialFeat:", spatialFeat)
        print ("histFeat:", histFeat)
        print ("hogFeat:", hogFeat)

        vehicleClassifier = Classifier(CLASSIFIER_FILE, args.debug)
        # Train the classifier
        vehicleClassifier.train(args.train_classifier, testSize, orientation, pxPerCell, cellPerBlock, hogChannel, spatialSize, histBins, spatialFeat, histFeat, hogFeat, colorspace)
        detectVehicles.vehicleClassifier = [vehicleClassifier]
    elif (args.classifiers is not None):
        classifiers = args.classifiers.split(',')
        activeClassifiers = []
        for classifier in classifiers:
            fileName = "%s/%s" % (classifier, CLASSIFIER_FILE)
            vehicleClassifier = Classifier(fileName, args.debug)
            vehicleClassifier.load()
            activeClassifiers.append(vehicleClassifier)

        # load the classifier
        detectVehicles.vehicleClassifier = activeClassifiers
    else:
        print ("No classifiers specified (-q CLASSIFIERS)\n")
        parser.print_help()
        sys.exit(0)


    # Process an input video
    if ((args.video_in is not None) and (args.video_out is not None)):
        # load video for processing
        outputVideo = args.video_out
        testVideo = VideoFileClip(args.video_in)
        output = testVideo.fl_image(detectVehicles)
        output.write_videofile(outputVideo, audio=False)


    # process a single input image
    if (args.single_image is not None):
        image = mpimg.imread(args.single_image)
        output, image, detectionBoxes, heat, heatThreshold = detectVehicles(image, standalone=True)

        windowImg = Classifier.overlayBoxes(image, detectionBoxes, color=(0, 0, 255), thickness=6) 
        top = np.concatenate((image, windowImg), axis=1)
        mid = np.concatenate((heat, heatThreshold), axis=1)  
        bot = np.concatenate((output, output), axis=1)                    

        result = np.concatenate((top, mid, bot), axis=0)
        plt.imshow(output)
        plt.show()

    # Process an input directory of images, testing a number of different classification paramters.
    if (args.image_directory is not None):
        for fileName in os.listdir(args.image_directory):
            if "output" in fileName:
                continue
            image = mpimg.imread("%s/%s" % (args.image_directory, fileName))
            result, image, detectionBoxes, heat, heatThreshold = detectVehicles(image, standalone=True)

            windowImg = Classifier.overlayBoxes(image, detectionBoxes, color=(0, 0, 255), thickness=6) 
            top = np.concatenate((image, windowImg), axis=1)
            mid = np.concatenate((heat, heatThreshold), axis=1)  
            bot = np.concatenate((result, result), axis=1)                    

            result = np.concatenate((top, mid, bot), axis=0)
            plt.imshow(result)
            plt.show()
