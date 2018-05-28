# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:43:15 2018

@author: Administrator
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

''' TOPIC 1.1 '''
#cv2.namedWindow('test',cv2.WINDOW_NORMAL)
#cv2.namedWindow('test',cv2.WINDOW_AUTOSIZE)
#img = cv2.imread('test2.jpg',0)
#cv2.imshow('test',img)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()


''' TOPIC 1.2 '''
#    img =  cv2.imread('test2.jpg')
#    #b,g,r = cv2.split(img)
#    #img2 = cv2.merge([r,g,b])
#    
#    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#
#    #plt.imshow(img2,cmap = 'gray',interpolation = 'bicubic')
#    plt.imshow(img2)
#    plt.xticks([])
#    plt.yticks([])
#    plt.show()


''' TOPIC 2.1 '''
#    print 'start'
#    cap = cv2.VideoCapture("rtsp://192.168.157.141:554/cam/realmonitor?channel=0&subtype=0&unicast=true&proto=Onvif")
#    print cap.isOpened()
#    while cap.isOpened:
#        ret, frame = cap.read()
#        
#        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#        
#        cv2.imshow('frame',gray)
#        if cv2.waitKey(1) & 0xff == ord('q'):
#            break
#    
#    cap.release()
#    cv2.destroyAllWindows()


''' TOPIC 2.2 '''
#    print 'start'
#    cap = cv2.VideoCapture("rtsp://192.168.157.141:554/cam/realmonitor?channel=0&subtype=0&unicast=true&proto=Onvif")
#    print cap.isOpened()
#    #fourcc = cv2.VideoWriter_fourcc(*'X264') for opencv3
#    fourcc = cv2.cv.CV_FOURCC(*'X264')
#    out = cv2.VideoWriter('output.avi',fourcc,25.0,(1920,1080))
#    
#    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
#    
#    while cap.isOpened:
#        ret, frame = cap.read()
#        if ret == True:
#            frame = cv2.flip(frame,0)
#            out.write(frame) #未写成功
#            cv2.imshow('frame',frame)
#            if cv2.waitKey(1) & 0xff == ord('q'):
#                break
#        else:
#            break
#    
#    cap.release()
#    out.release()
#    cv2.destroyAllWindows()


''' TOPIC 3 '''
#    cv2.namedWindow('draw',cv2.WINDOW_NORMAL)
#    img = np.zeros((512,512,3),np.uint8)
#    
#    cv2.line(img,(0,0),(511,511),(255,0,0),5)
#    
#    cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
#    
#    cv2.circle(img,(447,63),63,(0,0,255),-1)
#    
#    cv2.ellipse(img,(256,256),(100,50),0,0,180,(0,255,255),-1)
#    
#    pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
#    pts = pts.reshape((-1,1,2))
#    cv2.polylines(img,[pts],True,(255,255,0))
#    
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    cv2.putText(img,'OpenCV',(10,500),font,4,(255,255,255),2,cv2.CV_AA)
#    
#    cv2.imshow('draw',img)
#    
#    cv2.waitKey(0)
#    
#    cv2.destroyAllWindows()


''' TOPIC 3.1 '''
#    # to list all available events 
#    #events = [i for i in dir(cv2) if 'EVENT' in i]
#    #print events
#    img = np.zeros((512,512,3),np.uint8)
#    cv2.namedWindow('image')    
#    def draw_circle(event,x,y,flags,param):
#        if event == cv2.EVENT_LBUTTONDBLCLK:
#            cv2.circle(img,(x,y),100,(0,0,255),5)
#    
#    cv2.setMouseCallback('image',draw_circle)
#    
#    while(1):
#        cv2.imshow('image',img)
#        if cv2.waitKey(20) & 0xff == 27: #ASCII 27 is ESC
#            break
#    cv2.destroyAllWindows()


''' TOPIC 3.2 '''
#    drawing = False
#    mode = True # if True,draw rectangle. Press 'm' to toggle to curve
#    ix,iy = -1,-1
#
#    img = np.zeros((512,512,3),np.uint8)
#    cv2.namedWindow('image')    
#
#    def draw_circle(event,x,y,flags,param):
#        global ix,iy,drawing,mode,jx,jy
#        
#        if event == cv2.EVENT_LBUTTONDOWN:
#            drawing = True
#            ix,iy = x,y
#          
#        elif event == cv2.EVENT_MOUSEMOVE:
#            if drawing == True:
#                if mode == True:
#                    cv2.rectangle(img,(ix,iy),(x,y),(30,0,200),5)
#                else:
#                    cv2.circle(img,(x,y),5,(0,0,255),5)
#        elif event == cv2.EVENT_LBUTTONUP:
#            drawing = False
#            if mode == True:
#                cv2.rectangle(img,(ix,iy),(x,y),(30,0,200),-1)
#            else:
#                cv2.circle(img,(x,y),5,(0,0,255),5)
#        
#    cv2.setMouseCallback('image',draw_circle)
#    
#    while(1):
#        cv2.imshow('image',img)
#        k = cv2.waitKey(20) & 0xff
#        if k == ord('m'):
#            mode = not mode
#        elif k == 27:
#            break
#    cv2.destroyAllWindows()


''' TOPIC 3.3 '''
#    drawing = False
#    ix,iy = -1,-1
#    jx,jy = -1,-1
#
#    img = np.zeros((512,512,3),np.uint8)
#    cv2.namedWindow('image')    
#
#    def draw_rect(event,x,y,flags,param):
#        global ix,iy,drawing,jx,jy
#        
#        if event == cv2.EVENT_LBUTTONDOWN:
#            drawing = True
#            ix,iy = x,y
#            jx,jy = x,y
#        elif event == cv2.EVENT_MOUSEMOVE:
#            if drawing == True:
#                cv2.rectangle(img,(ix,iy),(jx,jy),(0,0,0),5)
#                cv2.rectangle(img,(ix,iy),(x,y),(30,0,200),5)
#                jx,jy = x,y
#                
#        elif event == cv2.EVENT_LBUTTONUP:
#            drawing = False
# 
#        
#    cv2.setMouseCallback('image',draw_rect)
#    
#    while(1):
#        cv2.imshow('image',img)
#        k = cv2.waitKey(20) & 0xff
#        if k == ord('m'):
#            mode = not mode
#        elif k == 27:
#            break
#    cv2.destroyAllWindows()


''' TOPIC 4 '''
#   def nothing(x):
#        pass
#    
#    img = np.zeros((300,512,3),np.uint8)
#    cv2.namedWindow('image')
#    
#    cv2.createTrackbar('R','image',0,255,nothing)
#    cv2.createTrackbar('G','image',0,255,nothing)
#    cv2.createTrackbar('B','image',0,255,nothing)
#    
#    switch = r'0 : OFF 1: ON'
#    cv2.createTrackbar(switch,'image',0,1,nothing)
#    
#    while(1):
#        cv2.imshow('image',img)
#        k = cv2.waitKey(1) & 0xFF
#        if k == 27:
#            break
#        
#        r = cv2.getTrackbarPos('R','image')
#        g = cv2.getTrackbarPos('G','image')
#        b = cv2.getTrackbarPos('B','image')
#        s = cv2.getTrackbarPos(switch,'image')
#        if s == 0:
#            img[:] = 0
#        else:
#            img[:] = [b,g,r]
#    
#    cv2.destroyAllWindows()

if __name__ == '__main__':
    pass