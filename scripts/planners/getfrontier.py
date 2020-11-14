#!/usr/bin/env python
"""
Code from https://github.com/hasauino/rrt_exploration/blob/master/scripts/getfrontier.py
"""

#--------Include modules---------------
from copy import copy
import rospy
from nav_msgs.msg import OccupancyGrid
import os

import numpy as np
import cv2

#-----------------------------------------------------

def getfrontier(mapData):
	data=mapData.data
	w=mapData.info.width
	h=mapData.info.height
	resolution=mapData.info.resolution
	Xstartx=mapData.info.origin.position.x
	Xstarty=mapData.info.origin.position.y
	 
	img = np.zeros((h, w, 1), np.uint8)
	
	for i in range(0,h):
		for j in range(0,w):
			if data[i*w+j]==100:
				img[i,j]=0
			elif data[i*w+j]==0:
				img[i,j]=255
			elif data[i*w+j]==-1:
				img[i,j]=255#205
	
	
	o=cv2.inRange(img,0,1)

	print("width {} height {}".format(w,h))
    # Edge detection
	dst = cv2.Canny(img,0,255)  
	os.chdir(os.path.dirname(__file__))
	# print(os.getcwd())
	# cv2.imwrite('image.jpg',img)
	
	#  Standard Hough Line Transform
	minLineLength = 2
	maxLineGap = 10
	lines = cv2.HoughLines(dst, 1, np.pi / 90, 200,minLineLength,maxLineGap)
	print("lines",lines)
	# Draw the lines
	all_pts=[]
	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = np.cos(theta)
			x0 = a * rho
			y0 = b * rho
			b = np.sin(theta)
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
			all_pts.append((pt1,pt2))
			cv2.line(img,pt1,pt2,(0,0,255),2)

	cv2.imwrite('houghlines3.jpg',img)

	# edges = cv2.Canny(img,0,255)
	# im2, contours, hierarchy = cv2.findContours(o,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(o, contours, -1, (255,255,255), 5)
	# o=cv2.bitwise_not(o) 
	# res = cv2.bitwise_and(o,edges)
	# #------------------------------

	# frontier=copy(res)
	# im2, contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(frontier, contours, -1, (255,255,255), 2)

	# im2, contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# print("contour",contours)
	# all_pts=[]
	# if len(contours)>0:
	# 	upto=len(contours)-1
	# 	i=0
	# 	maxx=0
	# 	maxind=0
		
	# 	for i in range(0,len(contours)):
	# 			cnt = contours[i]
	# 			M = cv2.moments(cnt)
	# 			cx = int(M['m10']/M['m00'])
	# 			cy = int(M['m01']/M['m00'])
	# 			xr=cx*resolution+Xstartx
	# 			yr=cy*resolution+Xstarty
	# 			pt=[np.array([xr,yr])]
	# 			if len(all_pts)>0:
	# 				all_pts=np.vstack([all_pts,pt])
	# 			else:
							
	# 				all_pts=pt
	print("frontier points",all_pts)
	return all_pts

