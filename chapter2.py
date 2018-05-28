# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:39:30 2018

@author: Administrator
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt



''' TOPIC 1.1 '''
#    img = cv2.imread('test1.jpg')
#    
#    # accessing one pixel (bad way)
#    px = img[100,100]
#    print px
#    
#    # accessing only blue channel of one pixel (bad way)
#    blue = img[100,100,0]
#    print blue
#    
#    # modify the pixel (bad way)
#    img[100,100] = [255,255,255]
#    print img[100,100]
#    
#    # Numpy is a optimized library for fast array calculations.
#    # So simply accessing each and every pixel values and modifying it will be
#    # very slow and it is discouraged.
#        
#    # accessing only red channel of one pixel (good way)
#    print img.item(10,10,2)
#    
#    img.itemset((10,10,2),100)
#    print img.item(10,10,2)
#    
#    # If image is grayscale, tuple returned contains only number of rows and 
#    # columns. So it is a good method to check if loaded image is grayscale 
#    # or color image.
#    print img.shape
#    
#    # total number of pixels
#    print img.size
#    
#    # image datatype
#    print img.dtype
#    
#    # ROI
#    roi = img[280:300,350:369] # uninclude the pos behind ':'
#    print roi.shape
#    img[0:20,0:19] = roi
#    
#    # split and merge
#    # cv2.split() is a costly operation (in terms of time). 
#    # So do it only if you need it. Otherwise go for Numpy indexing.
#    
#    b,g,r = cv2.split(img)
#    img = cv2.merge((b,g,r))
#    # or
#    b = img[:,:,0]
#    
#    # make all the red pixels to zero
#    img[:,:,2] = 0
    

''' TOPIC 1.2 '''
#    BLUE = [255,0,0]
#    img1 = cv2.imread('opencv_logo.png')
#    
#    replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
#    reflect= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
#    reflect101= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
#    wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
#    constant = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
#    
#    plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL'),plt.xticks([]),plt.yticks([])
#    plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE'),plt.xticks([]),plt.yticks([])
#    plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT'),plt.xticks([]),plt.yticks([])
#    plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101'),plt.xticks([]),plt.yticks([])
#    plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP'),plt.xticks([]),plt.yticks([])
#    plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT'),plt.xticks([]),plt.yticks([])
#   
#    plt.show()
    

''' TOPIC 2.1 '''
#    # differenct between OpenCV addition and Numpy addition
#    x = np.uint8([250])
#    y = np.uint8([10])
#    print cv2.add(x,y) # 250+10 = 260 => 255
#    print x+y # 250+10 = 260 % 256 = 4


''' TOPIC 2.2 ''' # not exercised,need pictures in same size
#    img1 = cv2.imread('test2.jpg')
#    img2 = cv2.imread('opencv_logo.png')
#
#    dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
#
#    cv2.imshow('dst',dst)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


''' TOPIC 2.3 '''
#    img1 = cv2.imread('opencv_logo.png')
#    img2 = cv2.imread('test2.jpg')
#    
#    rows,cols,channels = img1.shape
#    roi = img2[0:rows,0:cols]
#    
#    img1gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#    ret,mask = cv2.threshold(img1gray,10,255,cv2.THRESH_BINARY)
#    mask_inv = cv2.bitwise_not(mask)
#    
#    img2_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
#    img1_fg = cv2.bitwise_and(img1,img1,mask = mask)
#
#    dst = cv2.add(img2_bg,img1_fg)
#    img2[0:rows,0:cols] = dst
#    
#    cv2.imshow('res',img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


''' TOPIC 2.4 '''
#    img1 = cv2.imread('opencv_logo.png')
#    img2 = cv2.imread('test2.jpg')
#    
#    rows,cols,channels = img1.shape
#    roi = img2[0:rows,0:cols]
#    
#    img1gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#    ret,mask = cv2.threshold(img1gray,100,255,cv2.THRESH_BINARY_INV)
#    mask_inv = cv2.bitwise_not(mask)
#    
#    img2_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
#    img1_fg = cv2.bitwise_and(img1,img1,mask = mask)
#
#    dst = cv2.add(img2_bg,img1_fg)
#    img2[0:rows,0:cols] = dst
#    
#    cv2.imshow('res',img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


''' TOPIC 3.1 '''
#    img1 = cv2.imread('test2.jpg')
#    e1 = cv2.getTickCount()
#    for i in xrange(5,49,2):
#        img1 = cv2.medianBlur(img1,i)
#    e2 = cv2.getTickCount()
#    t = (e2-e1)/cv2.getTickFrequency()
#    
#    print t


''' TOPIC 3.2 '''
# check if optimization is enabled
#In [5]: cv2.useOptimized()
#Out[5]: True
#
#In [6]: %timeit res = cv2.medianBlur(img,49)
#10 loops, best of 3: 34.9 ms per loop
#
## Disable it
#In [7]: cv2.setUseOptimized(False)
#
#In [8]: cv2.useOptimized()
#Out[8]: False
#
#In [9]: %timeit res = cv2.medianBlur(img,49)
#10 loops, best of 3: 64.1 ms per loop


''' TOPIC 3.3 '''
# Python scalar operations are faster than Numpy scalar operations. So for 
# operations including one or two elements, Python scalar is better than Numpy
# arrays. Numpy takes advantage when size of array is a little bit bigger.

#In [10]: x = 5
#
#In [11]: %timeit y=x**2
#10000000 loops, best of 3: 73 ns per loop
#
#In [12]: %timeit y=x*x
#10000000 loops, best of 3: 58.3 ns per loop
#
#In [15]: z = np.uint8([5])
#
#In [17]: %timeit y=z*z
#1000000 loops, best of 3: 1.25 us per loop
#
#In [19]: %timeit y=np.square(z)
#1000000 loops, best of 3: 1.16 us per loop


''' TOPIC 3.4 '''
# Normally, OpenCV functions are faster than Numpy functions. So for same
# operation, OpenCV functions are preferred. But, there can be exceptions,
# especially when Numpy works with views instead of copies.

#In [35]: %timeit z = cv2.countNonZero(img)
#100000 loops, best of 3: 15.8 us per loop
#
#In [36]: %timeit z = np.count_nonzero(img)
#1000 loops, best of 3: 370 us per loop


''' TOPIC 3.5 ''' 
# There are several techniques and coding methods to exploit maximum performance
# of Python and Numpy. Only relevant ones are noted here and links are given to
# important sources. The main thing to be noted here is that, first try to
# implement the algorithm in a simple manner. Once it is working, profile it,
# find the bottlenecks and optimize them.
#
#        1. Avoid using loops in Python as far as possible, especially
#            double/triple loops etc. They are inherently slow.
#        2. Vectorize the algorithm/code to the maximum possible extent because
#            Numpy and OpenCV are optimized for vector operations.
#        3. Exploit the cache coherence.
#        4. Never make copies of array unless it is needed. Try to use views
#            instead. Array copying is a costly operation.
#
# Even after doing all these operations, if your code is still slow, or use of
# large loops are inevitable, use additional libraries like Cython to make it
# faster.
# Python Optimization Techniques:http://wiki.python.org/moin/PythonSpeed/PerformanceTips
# Scipy Lecture Notes :http://scipy-lectures.github.io/advanced/advanced_numpy/index.html#advanced-numpy
# Timing and Profiling in IPython:http://pynash.org/2013/03/06/timing-and-profiling.html

if __name__ == '__main__':
