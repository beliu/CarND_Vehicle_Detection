## Writeup 
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Image Data Preparation
Before I extract features from the images, I first wrote a function called `make_training_files()` that takes in a collection of datasets and combines them into a large list of image file paths and file names. The function separates the car from the non-car data and shuffles them based on time-sequence so that images that are very similar are not close to each other in sequence. You can also specify whether to augment the dataset, in which case a function called `augment()` will be called. The user can specify whether to perform rotation, flips, and brightness adjustment on the dataset. I also wrote a function called `extract_images()` that I used on the larger Udacity dataset. This function extracts the image of objects based on their pixel locations found in the Udacity dataset's CSV file and saves them to a folder. In the end, I found that I did not need to use either augmentation and the Udacity dataset to supplement the existing dataset.


### Feature Extraction, Including Histogram of Oriented Gradients (HOG)
#### 1. Explain how (and identify where in your code) you extracted image features from the training images.
I created a function called `extract_features()` to extract an unraveled vector of HOG features from an image and append those features to the image's spatial features and a histogram of the image's features in a particular color space. In `extract_features()` lines 17 to 52, the user supplies a dictionary of the settings for each type of feature extraction operation. Then each image from a list of images are read and a vector of features is created by concatenating the spatial, histogram, and HOG features together. Each new image feature vector is appended to a list called `features`. This step is in lines 52 to 54. The two images below show an example of the dominant gradients for an image in the HSV and YCrCb color space respectively.

![HOG HSV Space][image1]
![HOG YCrCb Space][image2]

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them). 
I created a function called `make_svc()` which takes in as input a dictionary of the parameters for each type of feature extraction. The function extracts the features from the images in the dataset by calling `extract_features()` and combines the feature vector of each image into a large array. I then use `sklearn.preprocessing.StandardScaler()` on this features array to normalize the data. This is done in lines 19 to 25 of the function `make_svc()`. The standardized features array is split into a training and validation set. I then use sklearn's `LinearSVC()` on the training dataset to train a classifier. After the classifier is done training, I calculate the model's accuracy on the validation set as a metric to determine whether the parameters I chose for feature extraction is good. This is done in lines 45 to 51 of `make_svc()`. 

#### 3. Explain how you settled on your final choice of HOG parameters.
My main metric for determining a set of HOG parameters is by looking at the model's accuracy score on the By experimenting with different parameters and color spaces, I saw that I got fairly high accuracy if I converted the image to either HSV or YCrCb space on all three image channels. For the other parameters, I stayed with the default parameters (`orientations=9`, `pixels_per_cell=8`, and `cells_per_block=2`).

### Sliding Window Search
#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
I implemented a sliding window search in the function `find_cars()`. This function allows the user to specify which portion of the image to search and what the scale of the search window is. It takes an image, converts it to the desired color space, and calculates the HOG features for each color channel without unraveling it. Now the program takes one scaled window at a time and slides it across the entire HOG feature array one image channel at a time. The window is 64 px by 64 px because this is the size of the images in the training set. We need these two to match so that the feature vectors are of equal length. Now, the HOG features in this window are unraveled into a vector. The window remains in position but we take the unraveled features of the next image channel and add it to the existing HOG features vector. Once all the channels the user specified are processed, we move the window to the next position. For this project, I kept `cells_per_step=2`, so that the window moves 2 cells each step. You can find this process in lines 48 to 64. 

If the user wants to use spatial and color histogram information, then the next step is to take the patch of image defined by the window, resize it to be (64, 64) and get its spatial and histogram features per channel. These features are appended to the HOG features and the entire vector is passed into the LinearSVC. If the classifier predicts that the sub-image is a car, it will output the top left and bottom right coordinates of a bounding box that surrounds the sub-image. This is done in lines 89 to 101 of `find_cars()`.

The entire process described above until the same-scaled window slides up and own the image. If there is more than one scale, we repeate the process with the new scale. Each time the classifier predicts that the sub-image found within the window is a car, it will add the coordinates of the bounding box corners to a list. Once all windows are completed, the list will contain all the corner coordinates of the bounding box surrounding positive detections.

![Multi-Scale Search Windows][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

