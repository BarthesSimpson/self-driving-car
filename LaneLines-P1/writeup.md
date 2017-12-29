**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "First Pass"
[image2]: ./test_images_output_better/solidWhiteCurve.jpg "Second Pass"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

First, I refactored some of the helper methods so they didn't contain side effects and so that the parameters for the whole pipeline could be tweaked in one central location via a settings dictionary.

My pipeline then consisted of the following steps:

- convert to grayscale
- apply a gaussian blur
- perform canny edge detection
- apply a clipping mask to isolate a region of interest relative to the camera perspective
- use a hough transform to group the detected edges into lines

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by grouping all the hough lines based on the sign (positive or negative) of their slope. For each group, I then averaged their slope and intercept to give a single line. Finally, I extended both lines to span the entire region of interest.

![Before Improvement][image1]
![After Improvement][image2]


### 2. Identify potential shortcomings with your current pipeline

My pipeline only works on roughly straight lines (no sharp curves). The roi is also hard coded, which would be a problem if the image resolution were to change. I originally wrote my roi as a function of the aspect ratio, but that made debugging harder so I reverted to a simpler implementation.

### 3. Suggest possible improvements to your pipeline

To be honest, this pipeline seems really flimsy. Since this is the first assignment, I don't yet know enough to suggest effective improvements, but at the very least I'd like to be able to detect curves instead of just straight lines. I believe the Hough method can be generalized to do this.