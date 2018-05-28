# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:30:20 2018

@author: Administrator
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

''' TOPIC 1.1 ''' # not exercised
# For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range 
# is [0,255]. Different softwares use different scales. So if you are comparing
# OpenCV values with them, you need to normalize these ranges.

#    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#    print flags
#
#    cap = cv2.VideoCapture(0)
#    
#    while(1):
#    
#        # Take each frame
#        _, frame = cap.read()
#    
#        # Convert BGR to HSV
#        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    
#        # define range of blue color in HSV
#        lower_blue = np.array([110,50,50])
#        upper_blue = np.array([130,255,255])
#    
#        # Threshold the HSV image to get only blue colors
#        mask = cv2.inRange(hsv, lower_green, upper_green)
#    
#        # Bitwise-AND mask and original image
#        res = cv2.bitwise_and(frame,frame, mask= mask)
#    
#        cv2.imshow('frame',frame)
#        cv2.imshow('mask',mask)
#        cv2.imshow('res',res)
#        k = cv2.waitKey(5) & 0xFF
#        if k == 27:
#            break
#    
#    cv2.destroyAllWindows()


''' TOPIC 1.2 ''' # not exercised
# example: to find the HSV value of Green
#   green = np.uint8([[[0,255,0]]])
#   hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#   print hsv_green

# get [[[60,255,255]]]

# Now you take [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper
# bound respectively. Apart from this method, you can use any image editing
# tools like GIMP or any online converters to find these values, but don’t
# forget to adjust the HSV ranges.


''' TOPIC 2.1 '''
#    img = cv2.imread('test3.jpg',0)
#    
#    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#    
#    titles = ['original image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#    images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
#    
#    for i in xrange(6):
#        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#        plt.title(titles[i])
#        plt.xticks([]),plt.yticks([])
#    
#    plt.show()


''' TOPIC 2.2 '''
#    img = cv2.imread('test4.jpg',0)
#    img = cv2.medianBlur(img,5)
#    
#    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#    
#    titles = ['Original Image','Global Thresholding(v=127)',
#              'Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
#    images = [img,th1,th2,th3]
#    
#    for i in xrange(4):
#        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#        plt.title(titles[i])
#        plt.xticks([]),plt.yticks([])
#        
#    plt.show()


''' TOPIC 2.3 '''
#    img = cv2.imread('test5.jpg',0)
#    
#    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    
#    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU) 
#    
#    blur = cv2.GaussianBlur(img,(5,5),0)
#    re3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    
#    images = [img,0,th1,img,0,th2,blur,0,th3]
#    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#          'Original Noisy Image','Histogram',"Otsu's Thresholding",
#          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
#    
#    for i in xrange(3):
#        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#        plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
#        
#        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#        plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
#
#        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#        plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
#    
#    plt.show()


''' TOPIC 2.4 '''
# Otsu's algorithm tries to find a threshold value (t) which minimizes the 
# weighted within-class variance given by the relation

#    img = cv2.imread('test5.jpg',0)
#    blur = cv2.GaussianBlur(img,(5,5),0)
#    
#    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
#    hist_norm = hist.ravel()/hist.max()
#    Q = hist_norm.cumsum()
#    
#    bins = np.arange(256)
#    
#    fn_min = np.inf
#    thresh= -1
#    
#    for i in xrange(1,255): # just 254 times iteration
#        p1,p2 = np.hsplit(hist_norm,[i])
#        q1,q2 = Q[i],Q[255]-Q[i]
#        b1,b2 = np.hsplit(bins,[i])
#        
#        m1,m2 = np.sum(p1*b1)/q1,np.sum(p2*b2)/q2
#        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
#        
#        fn = v1*q1 + v2*q2
#        if fn < fn_min:
#            fn_min = fn
#            thresh = i
#    
#    ret,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    print thresh,ret


''' TOPIC 3.1 ''' # not exercised
#scaling
# img = cv2.imread('test2.jpg')
# res = cv2.resize(img,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
#OR
# height,width = img.shape[:2]
# res = cv2.resize(img,(2*width,2*height),interpolation = cv2.INTER_CUBIC)


''' TOPIC 3.2 ''' 
#Translation
# Third argument of the cv2.warpAffine() function is the size of the output
# image, which should be in the form of (width, height). Remember width = 
# number of columns, and height = number of rows.

#    img = cv2.imread('test2.jpg',0)
#    rows,cols = img.shape
#    M = np.float32([[1,0,100],
#                    [0,1,50]])
#    dst = cv2.warpAffine(img,M,(cols,rows))
#    
#    cv2.imshow('img',dst)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 


''' TOPIC 3.3 '''
# rotation
#   img = cv2.imread('test2.jpg',0)
#   rows,cols = img.shape
#   
#   M = cv2.getRotationMatrix2D((cols/2,rows/2),90,0.5) #last param for scaling
#   dst = cv2.warpAffine(img,M,(cols,rows))
#   
#   cv2.imshow('img',dst)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows() 


''' TOPIC 3.4 '''
#Affine Transformation
#    img = cv2.imread('test2.jpg')
#    rows,cols,ch = img.shape
#    
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    
#    pts1 = np.float32([[50,50],[200,50],[50,200]])
#    pts2 = np.float32([[10,100],[200,50],[100,250]])
#    
#    M = cv2.getAffineTransform(pts1,pts2)
#    
#    dst = cv2.warpAffine(img,M,(cols,rows))
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Input')
#    plt.subplot(122),plt.imshow(dst),plt.title('Output')
#    plt.show()


''' TOPIC 3.5 ''' #not exercised
#    img = cv2.imread('sudokusmall.png')
#    rows,cols,ch = img.shape
#    
#    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
#    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
#    
#    M = cv2.getPerspectiveTransform(pts1,pts2)
#    
#    dst = cv2.warpPerspective(img,M,(300,300))
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Input')
#    plt.subplot(122),plt.imshow(dst),plt.title('Output')
#    plt.show()


''' TOPIC 4.1 '''
#2D Convolution: averaging filter
#    img = cv2.imread('opencv_logo.png')
#    kernel = np.ones((5,5),np.float32)/25
#    dst = cv2.filter2D(img,-1,kernel)
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
#    plt.xticks([]),plt.yticks([])
#    
#    plt.show()


''' TOPIC 4.2 '''
#Image blurring(Image Smoothing):Averaging 1/4
#    img = cv2.imread('opencv_logo.png')
#
#    blur = cv2.blur(img,(5,5))
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Original')
#    plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#    plt.xticks([]), plt.yticks([])
#    plt.show()


''' TOPIC 4.3 '''
#Image blurring(Image Smoothing):Gaussian Blurring 2/4
#    img = cv2.imread('opencv_logo.png')
#
#    blur = cv2.GaussianBlur(img,(5,5),0)
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Original')
#    plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#    plt.xticks([]), plt.yticks([])
#    plt.show()
    

''' TOPIC 4.4 '''
#Image blurring(Image Smoothing):Median Blurring 3/4

# This is highly effective against salt-and-pepper noise in the images. 
# Interesting thing is that, in the above filters, central element is a newly
# calculated value which may be a pixel value in the image or a new value.
# But in median blurring, central element is always replaced by some pixel
# value in the image. It reduces the noise effectively. Its kernel size should
# be a positive odd integer.

#    img = cv2.imread('opencv_logo.png')
#
#    blur = cv2.medianBlur(img,5)
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Original')
#    plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#    plt.xticks([]), plt.yticks([])
#    plt.show()


''' TOPIC 4.5 '''
#Image blurring(Image Smoothing):Bilateral Filtering 4/4

# Bilateral Filtering is highly effective in noise removal while keeping edges
# sharp.Bilateral filter also takes a gaussian filter in space, but one more
# gaussian filter which is a function of pixel difference. Gaussian function
# of space make sure only nearby pixels are considered for blurring while
# gaussian function of intensity difference make sure only those pixels with
# similar intensity to central pixel is considered for blurring. So it
# preserves the edges since pixels at edges will have large intensity variation.

#    img = cv2.imread('opencv_logo.png')
#
#    blur = cv2.bilateralFilter(img,9,75,75)
#    
#    plt.subplot(121),plt.imshow(img),plt.title('Original')
#    plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#    plt.xticks([]), plt.yticks([])
#    plt.show()


''' TOPIC 5 '''
# Morphological transformations are some simple operations based on the image
# shape. It is normally performed on binary images. It needs two inputs, one
# is our original image, second one is called structuring element or kernel
# which decides the nature of operation. Two basic morphological operators are
# Erosion and Dilation. Then its variant forms like Opening, Closing, Gradient
# etc also comes into play. 

''' TOPIC 5.1 '''
# Erosion
#    img = cv2.imread('j.png',0)
#    kernel = np.ones((5,5),np.uint8)
#    erosion = cv2.erode(img,kernel,iterations = 1)
#    
#    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(erosion,'gray'),plt.title('Eroded')
#    plt.xticks([]),plt.yticks([])
#
#    plt.show()


''' TOPIC 5.2 ''' 
# Dilation
# It is just opposite of erosion.

#    img = cv2.imread('j.png',0)
#    kernel = np.ones((5,5),np.uint8)
#    erosion = cv2.erode(img,kernel,iterations = 1)
#    dilation = cv2.dilate(img,kernel,iterations = 1)
#    
#    plt.subplot(131),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(132),plt.imshow(erosion,'gray'),plt.title('Eroded')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(133),plt.imshow(dilation,'gray'),plt.title('Dilated')
#    plt.xticks([]),plt.yticks([])
#    plt.show()


''' TOPIC 5.3 '''
# Opening
# Opening is just another name of erosion followed by dilation.

#    img = cv2.imread('j_open.png',0)
#    kernel = np.ones((5,5),np.uint8)
#    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
#    
#    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(opening,'gray'),plt.title('Opened')
#    plt.xticks([]),plt.yticks([])
#
#    plt.show()


''' TOPIC 5.4 '''
# Closing
# Closing is reverse of Opening, Dilation followed by Erosion. It is useful in
# closing small holes inside the foreground objects, or small black points on
# the object.

#    img = cv2.imread('j_close.png',0)
#    kernel = np.ones((5,5),np.uint8)
#    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
#    
#    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(closing,'gray'),plt.title('Closed')
#    plt.xticks([]),plt.yticks([])
#
#    plt.show()


''' TOPIC 5.5 ''' 
# Morphological Gradient
# It is the difference between dilation and erosion of an image.

#    img = cv2.imread('j.png',0)
#    kernel = np.ones((5,5),np.uint8)
#    
#    gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
#    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(gradient,'gray'),plt.title('Gradient')
#    plt.xticks([]),plt.yticks([])
#
#    plt.show()


''' TOPIC 5.6 ''' 
# Top Hat
# It is the difference between input image and Opening of the image. 

#    img = cv2.imread('j.png',0)
#    kernel = np.ones((9,9),np.uint8)
#    
#    gradient = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
#    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(gradient,'gray'),plt.title('Top Hat')
#    plt.xticks([]),plt.yticks([])
#
#    plt.show()


''' TOPIC 5.7 '''
# Black Hat
# It is the difference between the closing of the input image and input image.

#    img = cv2.imread('j.png',0)
#    kernel = np.ones((9,9),np.uint8)
#    
#    gradient = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
#    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
#    plt.xticks([]),plt.yticks([])
#    plt.subplot(122),plt.imshow(gradient,'gray'),plt.title('Black Hat')
#    plt.xticks([]),plt.yticks([])
#
#    plt.show()


''' TOPIC 5.8 '''
# We manually created a structuring elements in the previous examples with
# help of Numpy. It is rectangular shape. But in some cases, you may need
# elliptical/circular shaped kernels. So for this purpose, OpenCV has a
# function, cv2.getStructuringElement(). You just pass the shape and size of
# the kernel, you get the desired kernel.

# Rectangular Kernel
#>>> cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#array([[1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel
#>>> cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#array([[0, 0, 1, 0, 0],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel
#>>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#array([[0, 0, 1, 0, 0],
#       [0, 0, 1, 0, 0],
#       [1, 1, 1, 1, 1],
#       [0, 0, 1, 0, 0],
#       [0, 0, 1, 0, 0]], dtype=uint8)


''' TOPIC 6 '''
# Image Gradients
# OpenCV provides three types of gradient filters or High-pass filters,
# Sobel, Scharr and Laplacian. 

''' TOPIC 6.1 '''
# 1. Sobel and Scharr Derivatives

# Sobel operators is a joint Gausssian smoothing plus differentiation operation,
# so it is more resistant to noise. You can specify the direction of derivatives
# to be taken, vertical or horizontal (by the arguments, yorder and xorder
# respectively). You can also specify the size of kernel by the argument ksize.
# If ksize = -1, a 3x3 Scharr filter is used which gives better results than
# 3x3 Sobel filter. Please see the docs for kernels used.

# 2. Laplacian Derivatives
# It calculates the Laplacian of the image given by the relation,
# \Delta src = \frac{\partial ^2{src}}{\partial x^2} + \frac{\partial ^2{src}}{\partial y^2}
# where each derivative is found using Sobel derivatives. If ksize = 1, then following kernel
# is used for filtering: kernel = [ [ 0, 1, 0]
#                                   [ 1,-4, 1]
#                                   [ 0, 1, 0] ]

#    img = cv2.imread('test4.jpg',0)
#    
#    laplacian = cv2.Laplacian(img,cv2.CV_64F)
#    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#    
#    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#    plt.title('Original'), plt.xticks([]), plt.yticks([])
#    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
#    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#    
#    plt.show()


''' TOPIC 6.2 '''
# Positive Slope & Negative Slope
# In our last example, output datatype is cv2.CV_8U or np.uint8. But there is
# a slight problem with that. Black-to-White transition is taken as Positive
# slope (it has a positive value) while White-to-Black transition is taken as
# a Negative slope (It has negative value). So when you convert data to np.uint8,
# all negative slopes are made zero. In simple words, you miss that edge.

# If you want to detect both edges, better option is to keep the output datatype
# to some higher forms, like cv2.CV_16S, cv2.CV_64F etc, take its absolute value
# and then convert back to cv2.CV_8U. Below code demonstrates this procedure for
# a horizontal Sobel filter and difference in results.

#    img = cv2.imread('box.png',0)
#
#    # Output dtype = cv2.CV_8U
#    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
#    
#    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
#    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#    abs_sobel64f = np.absolute(sobelx64f)
#    sobel_8u = np.uint8(abs_sobel64f)
#    
#    plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
#    plt.title('Original'), plt.xticks([]), plt.yticks([])
#    plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
#    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
#    plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
#    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
#    
#    plt.show()


''' TOPIC 7 '''
# Canny Edge Detection
# Canny Edge Detection is a multi-stage algorithm and we will go through each stages.
# 1. Noise Reduction
#   Since edge detection is susceptible to noise in the image, first step is to
#   remove the noise in the image with a 5x5 Gaussian filter. 
# 2. Finding Intensity Gradient of the Image
#   Smoothened image is then filtered with a Sobel kernel in both horizontal 
#   and vertical direction to get first derivative in horizontal direction (G_x)
#   and vertical direction (G_y). From these two images, we can find edge gradient
#   and direction for each pixel as follows:
#
#       Edge\_Gradient \; (G) = \sqrt{G_x^2 + G_y^2}
#
#       Angle \; (\theta) = \tan^{-1} \bigg(\frac{G_y}{G_x}\bigg)
#
#   Gradient direction is always perpendicular to edges. It is rounded to one of 
#   four angles representing vertical, horizontal and two diagonal directions.
# 3. Non-maximum Suppression
#   After getting gradient magnitude and direction, a full scan of image is done
#   to remove any unwanted pixels which may not constitute the edge. For this,
#   at every pixel, pixel is checked if it is a local maximum in its neighborhood
#   in the direction of gradient.In short, the result you get is a binary image
#   with “thin edges”.
# 4. Hysteresis Thresholding
#   This stage decides which are all edges are really edges and which are not.
#   For this, we need two threshold values, minVal and maxVal. Any edges with
#   intensity gradient more than maxVal are sure to be edges and those below 
#   minVal are sure to be non-edges, so discarded. Those who lie between these
#   two thresholds are classified edges or non-edges based on their connectivity. 
#   If they are connected to “sure-edge” pixels, they are considered to be part
#   of edges. Otherwise, they are also discarded. This stage also removes small
#   pixels noises on the assumption that edges are long lines.So what we finally
#   get is strong edges in the image.

# Canny Edge Detection in OpenCV
# OpenCV puts all the above in single function, cv2.Canny(). We will see how to
# use it. First argument is our input image. Second argument is output edge map.
# Third and Forth arguments are our minVal and maxVal respectively. Fifth argument
# is apertureSize. It is the size of Sobel kernel used for find image gradients. 
# By default it is 3. Last argument is L2gradient which specifies the equation
# for finding gradient magnitude. If it is True, it uses the equation mentioned
# above which is more accurate, otherwise it uses this function: Edge\_Gradient \; (G) = |G_x| + |G_y|.
# By default, it is False.

#    img = cv2.imread('test2.jpg',0)
#    edges = cv2.Canny(img,100,200,apertureSize = 3,L2gradient=True)
#    
#    plt.subplot(121),plt.imshow(img,cmap = 'gray')
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#    
#    plt.show()


''' TOPIC 8 '''
# Image Pyramids 
# Image Pyramids is a set of images(the same image) with different resolution and.
# There are two kinds of Image Pyramids. 
#   1) Gaussian Pyramid 
#   2) Laplacian Pyramids

''' TOPIC 8.1 '''
# Higher level (Low resolution) in a Gaussian Pyramid is formed by removing 
# consecutive rows and columns in Lower level (higher resolution) image. Then 
# each pixel in higher level is formed by the contribution from 5 pixels in 
# underlying level with gaussian weights. By doing so, a M \times N image becomes
# M/2 \times N/2 image. So area reduces to one-fourth of original area. It is
# called an Octave. The same pattern continues as we go upper in pyramid (ie,
# resolution decreases). Similarly while expanding, area becomes 4 times in each 
# level. We can find Gaussian pyramids using cv2.pyrDown() and cv2.pyrUp() functions.

#    img = cv2.imread('test2.jpg')
#    pyr2 = cv2.pyrDown(img)
#    pyr3 = cv2.pyrDown(pyr2)
#    pyr4 = cv2.pyrDown(pyr3)
#    
#    cv2.imshow('Level 1',img)
#    cv2.imshow('Level 2',pyr2)
#    cv2.imshow('Level 3',pyr3)
#    cv2.imshow('Level 4',pyr4)
#    
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

''' TOPIC 8.2 ''' #exercise it by self 

'''易错点：使用pyrDown的图片最好是4的整数倍，如果不是那么采用pyrUp后图片的尺寸与同一级
的down尺寸可能相差1。导致subtract报错'''

# Laplacian Pyramids
# A level in Laplacian Pyramid is formed by the difference between that level 
# in Gaussian Pyramid and expanded version of its upper level in Gaussian Pyramid.

#    img = cv2.imread('test2.jpg')
#    pyr2 = cv2.pyrDown(img)
#    pyr3 = cv2.pyrDown(pyr2)
#    pyr4 = cv2.pyrDown(pyr3)
#    
#    
#    tmp1 = cv2.pyrUp(pyr2)
#    tmp2 = cv2.pyrUp(pyr3)
#    tmp3 = cv2.pyrUp(pyr4)
#
#    Lap1 = cv2.subtract(img,tmp1)
#    Lap2 = cv2.subtract(pyr2,tmp2)
#    Lap3 = cv2.subtract(pyr3,tmp3)
#
#    cv2.imshow('Level 1',Lap1)
#    cv2.imshow('Level 2',Lap2)
#    cv2.imshow('Level 3',Lap3)
#  
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

''' TOPIC 8.3 '''
# Image Blending using Pyramids
# For details:http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/imagemosaic.html

#    A = cv2.imread('apple.jpg')
#    B = cv2.imread('orange.jpg')
#    
#    # generate Gaussian pyramid for A
#    G = A.copy()
#    gpA = [G]
#    for i in xrange(6):
#        G = cv2.pyrDown(G)
#        gpA.append(G)
#    
#    # generate Gaussian pyramid for B
#    G = B.copy()
#    gpB = [G]
#    for i in xrange(6):
#        G = cv2.pyrDown(G)
#        gpB.append(G)
#    
#    # generate Laplacian Pyramid for A
#    lpA = [gpA[5]]
#    for i in xrange(5,0,-1):
#        GE = cv2.pyrUp(gpA[i])
#        L = cv2.subtract(gpA[i-1],GE)
#        lpA.append(L)
#    
#    # generate Laplacian Pyramid for B
#    lpB = [gpB[5]]
#    for i in xrange(5,0,-1):
#        GE = cv2.pyrUp(gpB[i])
#        L = cv2.subtract(gpB[i-1],GE)
#        lpB.append(L)
#    
#    # Now add left and right halves of images in each level
#    LS = []
#    for la,lb in zip(lpA,lpB):
#        rows,cols,dpt = la.shape
#        ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
#        LS.append(ls)
#    
#    # now reconstruct
#    ls_ = LS[0]
#    for i in xrange(1,6):
#        ls_ = cv2.pyrUp(ls_)
#        ls_ = cv2.add(ls_, LS[i])
#    
#    # image with direct connecting each half
#    real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
#    
#    cv2.imwrite('Pyramid_blending2.jpg',ls_)
#    cv2.imwrite('Direct_blending.jpg',real)

if __name__ == '__main__':
    pass
    