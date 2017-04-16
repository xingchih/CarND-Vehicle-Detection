


#Vehicle Detection Project

### Xingchi He


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* It is important to normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Write the video pipeline function and run the pipeline on a video stream (test_video.mp4 and project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[example_car_not_car]: ./output_images/example_car_not_car.png
[example_hog_feagure]: ./output_images/example_hog_feagure.png
[sliding_window]: ./output_images/sliding_window.png
[multi_scale_sliding_window]: ./output_images/multi_scale_sliding_window.png
[testimg1]: ./output_images/testimg1.png
[testimg3]: ./output_images/testimg3.png
[testimg4]: ./output_images/testimg4.png
[testimg5]: ./output_images/testimg5.png
[bboxes_and_heat_1]: ./output_images/bboxes_and_heat_frame_1000.png
[bboxes_and_heat_2]: ./output_images/bboxes_and_heat_frame_1001.png
[bboxes_and_heat_3]: ./output_images/bboxes_and_heat_frame_1002.png
[bboxes_and_heat_4]: ./output_images/bboxes_and_heat_frame_1003.png
[bboxes_and_heat_5]: ./output_images/bboxes_and_heat_frame_1004.png
[bboxes_and_heat_6]: ./output_images/bboxes_and_heat_frame_1005.png
[labels_map]: ./output_images/labels_map_frame_1000.png
[output_bboxes]: ./output_images/output_bboxes_frame_1005.png
[project_video]: ./project_video_vehicle_detection.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell # of the IPython notebook`CarND_Vehicle_Detection.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][example_car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][example_hog_feagure]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as 
* number of HOG orietations
* Pixels per cell
* Cells per block
as well as different HOG channels. The final set of HOG parameters are:

| Parameter       | Value| 
|-----------------|---|
| Orientation #   | 9 | 
| Pixels per cell | 8 | 
| Cell per block  | 2 | 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features and color features.

I trained a linear SVM using `LinearSVC` in `sklearn.svm`. The feature vector has a length of 6108, and consists of HOG features, spatial color features, and color histogram. The total number of training images is 14208 and that of test images is 3552. It took about 16 seconds to train the SVC. The test accuracy is 99.24%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The first try was just to use a fixed sized window and slide it across the entire image

![alt text][sliding_window]

As pointed out in the class video, we only need to search the bottom half of the image because that is the region where cars can appear. Furthermore, I use three window sizes and assign each of them a specific y range to search, as illustrated below.
![alt text][multi_scale_sliding_window]

Blue squares are 64x64 and are intended to match cars that are furthur ahead of our car, as they will appear smaller. Cars that are closer will appear larger in the image, and we use green squares (96x96, 1.5x size) and red squares (128x128, 2x size) to detect them.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][testimg1]
![alt text][testimg3]
![alt text][testimg4]
![alt text][testimg5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_vehicle_detection_3.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Series of Images
Here are six frames and their corresponding heatmaps:
Frame 1000
![alt text][bboxes_and_heat_1]
Frame 1001
![alt text][bboxes_and_heat_2]
Frame 1002
![alt text][bboxes_and_heat_3]
Frame 1003
![alt text][bboxes_and_heat_4]
Frame 1004
![alt text][bboxes_and_heat_5]
Frame 1005
![alt text][bboxes_and_heat_6]

There are a few false positives in these images.

#### Integrated Heatmap
Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][labels_map]

The false positives are removed and the two cars are correctly detected.

#### Final Bounding Box 
Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][output_bboxes]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

