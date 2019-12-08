# Detecting Edges on image using convex optimization

`C++` Qt implementation of edge detection algorithms.

## Bachelor Research (Thesis)

The main idea of my Bachelor research work was to create a method to detect edges of cells using convex optimization, namely **Alternating Direction Method of Multipliers (ADMM)** (based on works of [S. Boyd](https://web.stanford.edu/~boyd/)) from a single image of the cell made by microscope.

**Reflections could be can be of two types:** 
- [ ] reflections have almost monotone color
- [ ] the color around edges varies smoothly. 

However, in the case where only a single image is given, reflection detection becomes much more challenging.

At first step I use **Canny edge detector** ([A Computational Approach to Edge Detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.3300&rep=rep1&type=pdf), J. Canny, 1986) to a given color image with reflections to detect all edges on the picture. Then was developed an algorithm for extracting non-reflection edges from the initial edge image by solving a certain convex optimization problem. And finally, the extracted image with edges is binarized by a simple thresholding operation. 

This algorithm can help doctors better detect cancer cells in [pictures made by microscope](https://github.com/ElizaLo/Edge-Detecting-Of-Reflections-On-Single-Image/tree/master/Dataset%20of%20Cells) because as we know cells mostly have monotone or smooth edge. As a result, it will be possible to detect the disease at the early stages and save a lot of human lives.

All this was implemented in **C ++**, without using any additional libraries. Since I was working with images, my task was to write the library for operations on matrices and tensors myself. Another difficulty of this work was the acceleration of the created algorithm using various optimizations such as [**efficient projections onto the _l<sub>1</sub>_-Ball for learning in high dimensions**](https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf).

I tested this method on a large number of photos of cells made under a microscope and made sure that the proposed method works and it can be used in medicine.


## **Available algorithms:**

 - **Created Algorithn using Convex Optimization**
 - [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
 - [Sobel edge detector](https://en.wikipedia.org/wiki/Sobel_operator)
 - [Prewitt edge detector](https://en.wikipedia.org/wiki/Prewitt_operator)
 - [Roberts cross](https://en.wikipedia.org/wiki/Roberts_cross)
 - Scharr operator
 
## Results
 
The project includes GUI for viewing results.

| | Photo | Cell image|
|:---:|:---:|:---:|
|**Original image**|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Original%20img%201.png" width="1106" height="391">|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Cell%20original.png" width="1106" height="391">|
|**Canny edge detector**|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Canny%20img%201.png" width="1106" height="391">|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Canny%20Cell%20.png" width="1106" height="391">|
|**Sobel edge detector**|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Sobel%20img%201.png" width="1106" height="391">|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Sobel%20Cell.png" width="1106" height="391">|
|**Prewitt edge detector**|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Prewitt%20img%201.png" width="1106" height="391">|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Prewitt%20Cell.png" width="1106" height="391">|
|**Roberts edge detector**|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Roberts%20img%201.png" width="1106" height="391">|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Roberts%20Cell.png" width="1106" height="391">|
|**Scharr edge detector**|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Scharr%20img%201.png" width="1106" height="391">|<img src="https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/blob/master/img/Scharr%20Cell.png" width="1106" height="391">|

[More images](https://github.com/ElizaLo/Detecting-Edges-on-Image-using-Convex-Optimization/tree/master/img) to compare.
  
## Requirements  

- `C++ 14`
 
## Used articles:

 - [Efficient Projections onto the l1-Ball for Learning in High Dimensions](https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf)
 - [Detecting Edges of Reflections from a Single Image Via CONVEX OPTIMIZATION](https://github.com/ElizaLo/Edge-detecting-of-reflections/blob/master/DETECTING%20EDGES%20OF%20REFLECTIONS%20FROM%20A%20SINGLE%20IMAGE.pdf)
 - [Alternating Direction Method of Multipliers (ADMM)](http://stanford.edu/~boyd/admm.html)
