# CannyEdgePytorch

A simple implementation of the Canny Edge Detection Algorithm (currently without hysteresis).

This project was implemented with PyTorch to take advantage of the parallelization of convolutions.

The original image:
<img src="https://github.com/DCurro/CannyEdgePytorch/blob/master/fb_profile.jpg" width="250">

Finding the gradient magnitude:
<img src="https://github.com/DCurro/CannyEdgePytorch/blob/master/gradient_magnitude.png" width="250">

Early thresholding (to show that edge thingging matters):
<img src="https://github.com/DCurro/CannyEdgePytorch/blob/master/thresholded.png" width="250">

And finally, the image after non-maximum supressions:
<img src="https://github.com/DCurro/CannyEdgePytorch/blob/master/thin_edges.png" width="250">
