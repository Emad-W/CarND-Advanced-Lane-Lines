## Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./images_writeup/straight_lines2_result.jpg)

In this project, the goal is to write a software pipeline using more advanced techniques to identify lane lines on the road.  

Firstly we will need to calibrate the camera and extract the calibration information and then use this information to correct camera distorations for the input images.

Afterwards will do image transformation and change image prespective to get the birds eye view to detect laneLines and compute the lane curvature and the Bias from center.


The project is built on the environment of [CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) that can be used as a starting point. Also it's needed to cover all of the [rubric points](https://review.udacity.com/#!/rubrics/1966/view) for this project.



---

Project Steps:
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  


Examples of the output from each stage of the pipeline are saved in the folder called `output_images`, Also The video called `project_video.mp4` is the video that the pipeline should work well on.  

Access [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points from here.

---

## **Advanced Lane Finding Project**


[//]: # (Image References)

[image0]: ./test_images/straight_lines2.jpg "Original"
[image1]: ./images_writeup/straight_lines2_undistorted.jpg "Undistorted"
[image2]: ./images_writeup/straight_lines2_Warped.jpg "Road Transformed"
[image3]: ./images_writeup/straight_lines2_binaryMask.jpg "Binary Example"
[image4]: ./images_writeup/straight_lines2_Output_Mask.jpg "Fit Visual"
[image5]: ./images_writeup/straight_lines2_result.jpg "Output"
[video1]: ./project_video.mp4 "Video"



### Here we discuss project steps and describe how they are implemented:  

---


### Camera Calibration

#### Computing the camera matrix and distortion coefficients. 

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".  
Main functions used are:
```python
    
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        imgpoints.append(corners)
        objpoints.append(objp)
        
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        
        cv2.imwrite((folder_out+fname),img)


```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

These `objpoints` and `imgpoints` are saved as'.pickle' files for later usage in the main pipeline without running CameraCalibration everytime.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function as follows:
```python
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    undist = cv2.undistort(img, mtx, dist, None, mtx)

```

Here is the obtained result: 

![alt text][image1]


---
### Pipeline for single images:

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image0]

Firstly we load the `objpoints` and `imgpoints` saved from the previous step, use them with `cv2.calibrateCamera()` function and `cv2.undistort()` to obtain the corrected image:
![alt text][image1]

#### 2. Color transforms, gradients for thresholded binary image. 

Next, By using a combination of S color chanel and magnitude gradient thresholds to generate a binary image as in function `binary_thresh()`as follows:
```python

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Apply each of the thresholding functions0
    mag_binary = mag_threshold(image, sobel_kernel=sobel_kernel, m_thresh=mag_thresh)
    
    color_binary = np.zeros_like(mag_binary)
    
    color_binary[((mag_binary == 1) | (s_binary == 1))] = 1

```

Here's an example of my output after applying this step.

![alt text][image3]

#### 3. Perspective transformation.

The code for perspective transform is by using hardcode four fixed source points mapped to another destination points in the following manner:

```python
# Warp and prespectie transform
src = np.float32([[481,520],[799,520],[213,690],[1065,690]])
dst = np.float32([[214,520],[1065,520],[214,690],[1065,690]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 481, 520      | 214, 520        | 
| 799, 520      | 1065, 520      |
| 213, 690     | 214, 690      |
| 1065, 690      | 1065, 690      |

this transform was verified by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image as shown hereunder.

![alt text][image2]

#### 4. Identifiying lane-line pixels and fitting their positions with a polynomial

After extracting lane lines as show above, identifying lane line and fitting the 2nd order polynomial to detect the lane lines and the area in between, this part is addressed through `fit_polynomial()` where the this function uses the lists that contain the pixels definning the lane lines that are generated through the 2nd function `find_lane_pixels()`, and then it curve fit this lanes in a 2nd order polynomial and then plot this polynomial and report back the 3 values for each lane line, the output of this step is then visualized in a mask as follows :

![alt text][image4]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

Using the polynomial values obtained, we can calculate the radius of curvature at certain point on the curve as follows:
```python

    left_curverad = 0  ## Implement the calculation of the left line here
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = 0  ## Implement the calculation of the right line here
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
```
Afterwards, Bias is calculated by back plotting points on the image and calculating the difference of x-axis value between left, right lanes with respect to the middle assuming that the camera is in the center of the car. 
```python
    ### Calculate Bias from center
    
    middle = Fit.binary_warped.shape[1] / 2
    
    dist_left = (middle - left_fitx[Fit.binary_warped.shape[0]-1])
    
    dist_right = (right_fitx[Fit.binary_warped.shape[0]-1] - middle)
    diff=(dist_left - dist_right) * xm_per_pix
    
```


#### 6. Final result.

Merging all the previous steps and annotating it with the result of Radius curvature and Car Bias, Here is an example of a final output on a test image:

![alt text][image5]

---

### Pipeline (video)

#### Final video output.

By applying the whole steps on a frame by frame basis and taking into consideration sanity checks and previous information over frames and ensuring the correctness of the lane detections as well as adding some additional robustness check. Here's a [link to my video result](./test_videos_output//project_video.mp4)

---

### Discussion

#### Problems / issues and future work

In this sample video, the main struggle was when the system can no longer identify lane lines from the background as the road is so bright such that our thresholding method fails to identify lane lines.

Other issue is when there is a sharp turn or when one of the lane lines is not included in the frame and it has to be assumed to match the other lane line.

These first problem could be addressed by implementing an automatic function that can easily identify the amount of illuminated pixels in the background and apply a proper threshold to compensate the higher value of illumination to make the algorithm smarter in tough light and background varying situations.

For the 2nd issue, implementing a function to check on lane by lane basis and calculate if the other lane should be expected within the image or out of camera boundaries would be a good solution, and will lead us to take corrective action to compensate over this limitation in our equipment.


---

Thanks for spending some time checking our work.
---
