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
Below is an example of the car detection function `find_cars()` on a three test images.

![Test Image 1][image4]
![Test Image 2][image4]
![Test Image 3][image4]

---

### Video Implementation
#### 1. Provide a link to your final video output.  
Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The function `process_image()` takes a frame (image) from the video and applies a multi-scale sliding window search method to detect cars in the frame. Through experimentation, I found that the window scale factors of 0.5, 1.25, and 1.5 applied to the y-regions of (400, 450), (400, 600), and (400 656) respectively produced the highest true-positive detection rate and the lowest false-positive detection rate. So, for each window scale, I slide the window in the specified region of the frame and get the bounding boxes around the detections. I repeat with the remaining window scales and I combine all the bounding boxes in total. This is done in lines 21 to 31 of `process_image()`. Afterwards, I convert the bounding boxes into heat maps by looping through the area encapsulated by each bounding box and incrementing the pixel values in those areas. 

In order to detect cars with higher confidence and to reject false positives, I accumulate heat maps across several frames. The specific number of frames can be adjusted but I found that 5 to 6 frames works well. For debugging purposes, I store a record of each frame's heat map to a list such that the i-th index of the list has a heat map corresponding to the i-th frame's detection. Below are some examples of heatmaps corresponding to subsequent frames from the video. Note that the images show the detected bounding boxes that result in the heatmaps shown.

![Heat Maps of Frames from Video][image5]

In order to convert the heat maps into separate, labeled entities, I use SciPy's `scipy.ndimage.measurements.label` function. I encapsulate the `label` function into one of my own called `make_labeled_boxes()` which just converts the output of `label` into a dictionary. The dictionary's keys are the label numbers and the values are the bounding box coordinates that surround the islands of heat in the heat map. So, after allowing the heat map from detected boxes to build up over 5 frames, I apply `make_labeled_boxes()` using a predefined threshold to return a dictionary of labeled boxes. If there are previous labeled boxes from past frames, I compare how much the current and previous ones overlap in lines 126 to 134 of `process_image()`. Supposing that the overlap ratio is above a predfined number, I conclude that the newly create labeled box is for the same object as the previous labeled box. Then I draw the box on the image.

#### 3. Describe how to track detected vehicles.
In lines 71 to 88 of `process_image()`, I look at the bounding boxes around the objects I detected from applying `make_labeled_boxes()` to a heat map. Since I believe the car is in those bounding boxes, over the next few frames I narrow my search to just within those boxes. I pass the pixel positions of the bounding boxes to `find_car()` as well as a different set of scaled windows that I found work better for subimages. Otherwise, the function works exactly the same as it does on the whole image, i.e. applying a multi-scaled sliding window search. Similar to object detection for the whole frame, we accumulate a few frames worth of heat maps for the subimages and then apply a threshold to get a final high-confidence detection. 

One benefit of looking just within the bounding boxes for detection instead of the entire image is that it makes the processing time shorter. Another benefit is that if there are no or few detections in the subimage, then we can say we have a false positive and remove the bounding box from the image. However, it also means that we are not detecting new objects that may appear elsewhere. Therefore, after a certain number of frames - 6 frames in this case - we go back to doing a sliding window search on the whole image. This process of searching the whole image, finding high-confidence detection objects, and then tracking only those objects, repeats for the entirety of the video. 

#### 4. Show some bounding boxes around detected objects
Within `process_image()`, I included an option to show the bounding boxes detected in a frame. The image below on the left shows detection boxes found during a sliding window search, which are depicted as thinner light-gold boxes. The image in the middle shows a resulting high-confidence box shown in a thick blue line that encapsulates an area where the detection boxes have added to surpass a defined threshold. The image on the right shows the detection boxes when we are just looking at subimages. Note that the search area is defined by the box in black line surrounding the objects, the purple boxes are the positive detection boxes from the sliding window search, and the blue line box once again encapsulates the high-confidence heat map detection made from the purple boxes.

![Detection Boxes][image]
![High Confidence Detection][image]
![Subimage Detection Boxes][image]

---

### Discussion
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I would say that the HOG LinearSVC is far from perfect. It has a much harder time detecting the white car consistently compared to the black cars. Also, it has a much higher chance of giving a false detection in shadowy areas. From previous projects, I noticed that an HSV color space helps with lighting and brightness but when I tried to use that color space for this project, I found that it did not work as well as YCrCb. I might consider combining HOG feature vectors from both color spaces together and then seeing if the combined feature vector can help with classification.

Another issue I noticed is that my pipeline is trouble detecting faster moving objects. Since it takes a few frames to be certain whether some object is a true positive, the fast-moving object in question may have already moved out of position from the time we first detect it to when we want to verify if it is a true detection. One way to mitigate this is to use motion planning to predict motions of objects at different places in the image and average the detections differently based on their speed. For example, we may want to add up detection boxes across more frames for slower moving objects but we may want to lower the number of frames for faster moving ones.

