#!/usr/bin/env python3
import pstats

import rospy
import numpy as np
import utils
import threading, time, collections, heapq, itertools, math, os
from itertools import count
from utils import Circle, Path, SearchNode, SearchNodeTree, TreeNode

from geometry_msgs.msg import PoseStamped, PolygonStamped, Point32
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Header
from nav_msgs.srv import GetMap

import cProfile

class DefinedWaypoints():
	def __init__(self):
		self.pub = rospy.Publisher("/defined_waypoints", MarkerArray, queue_size=1)
		self.waypointlist = MarkerArray()

		# 해당 좌표 리스트들은 향후 특정 파일에서 불러오도록 수정해야 함
		self.pointlist = [
			[3.764, 0.046, 0.0],
			[-37.097, 0.373, 0.0],
			[-52.644, 0.643, 0.0],
			[-52.49, 10.07, 0.0],
			[-36.878, 9.905, 0.0],
			[-36.285, 29.977, 0.0],
			[-19.955, 29.829, 0.0],
			[-19.93, 20.015, 0.0],
			[3.80,20.11, 0.0],
			[5.032, 45.84, 0.0],
			[-19.35, 45.75, 0.0],
			[-35.92, 60.98, 0.0],
			[-51.55, 61.35, 0.0]
		]

		for idx, point in enumerate(self.pointlist):
			self.add_waypoint(idx, point)

		self.map = None
		self.get_omap()

		print(self.map)


	def is_passable(self): # Check each node(waypoint) can be connected (no obstacle between each waypoints)
		pass

	def add_waypoint(self, idx, point):
		waypoint = Marker()

		waypoint.ns = str(idx)  # Marker namespace
		waypoint.id = idx  # Marker id value, no duplicates
		waypoint.text = str(idx)  # Marker namespace

		waypoint.type = 8  # line strip
		waypoint.lifetime = rospy.Duration.from_sec(100)
		waypoint.header = self.make_header("map")

		waypoint.action = 0 # ?

		# Set waypoint size
		waypoint.scale.x = 0.5
		waypoint.scale.y = 0.5
		waypoint.scale.z = 0.1

		# Set waypoint color, alpha
		waypoint.color.r = 0.85
		waypoint.color.g = 0.25
		waypoint.color.b = 0.25
		waypoint.color.a = 1.0

		pt = Point32()
		pt.x = point[0]
		pt.y = point[1]
		pt.z = point[2]
		waypoint.points.append(pt)
		self.waypointlist.markers.append(waypoint)

	def make_header(self, frame_id, stamp=None):
		if stamp == None:
			stamp = rospy.Time.now()
		header = Header()
		header.stamp = stamp
		header.frame_id = frame_id
		return header

	def get_omap(self):
		map_service_name = rospy.get_param("~static_map", "static_map")
		print("Fetching map from service: ", map_service_name)
		rospy.wait_for_service(map_service_name)
		map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
		# self.map = utils.Map(map_msg)
		self.map = map_msg
		self.map_initialized = True
		print("Finished loading map")

	def publish(self):
		print("publish")
		self.pub.publish(self.waypointlist)

class Node():
	f_score = 0
	g_score = 0
	h_score = 0
	parentnode = None

if __name__=="__main__":
	rospy.init_node("trajectory_search")
	dw = DefinedWaypoints()

	nodelist = []

	node1 = Node()
	node1.g_score = 2
	node2 = Node()
	node2.g_score = 55

	nodelist.append(node1)
	nodelist.append(node2)

	for node in nodelist:
		print(node.g_score)

	while True:
		dw.publish()
	pf = FindTrajectory()
	rospy.spin()