# Edge-detecting-of-reflections

Bachelor Research

My Bachelor research work is based on pattern recognition and edge detection. The main idea of this method is to detect edges of reflections using convex optimization, namely alternating direction method of multipliers (ADMM) from a single image. 

Reflections could be can be of two types: reflections have almost monotone color or color around edges varies smoothly.

First of all I use Canny edge detector to a given color image with reflections to detect all edges on the picture. Next, we extract reflection edges from the initial edge image by solving a certain convex optimization problem. And Finally, the extracted image with edges is binarized by a simple thresholding operation. 

I tested this method on a large number of photos of cells made under a microscope and made sure that the proposed method works and it can be used in medicine

Available algorithms:

 - Canny edge detector
 - Sobel edge detector
 - Prewitt edge detector
 - Roberts cross
 - Scharr operator
