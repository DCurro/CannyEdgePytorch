# CannyEdgePytorch

A simple implementation of the Canny Edge Detection Algorithm (currently without hysteresis).

This project was implemented with PyTorch to take advantage of the parallelization of convolutions.

The original image:
![alt text](https://github.com/DCurro/CannyEdgePytorch/blob/master/fb_profile.jpg)

Finding the gradient magnitude:
![alt text](https://github.com/DCurro/CannyEdgePytorch/blob/master/gradient_magnitude.png)

Early thresholding (to show that edge thingging matters):
![alt text](https://github.com/DCurro/CannyEdgePytorch/blob/master/thresholded.png)

And finally, the image after non-maximum supressions:
![alt text](https://github.com/DCurro/CannyEdgePytorch/blob/master/thin_edges.png)
