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
from pstats import SortKey

import matplotlib.pyplot as plt


class HeuristicSearch(object):
	""" Abstract class for heuristic search on the provided set of cost/admissibility/heuristic/neighbor functions """
	def __init__(self, make_tree=False):
		self.make_tree = make_tree

	def reset(self, start_state):
		# Frontier is a priority queue.
		self.closed_set = []
		self.frontier = []
		self.found_paths = []
		self.goal_state = self.goal()
		self.step_count = 0
		if self.make_tree:
			self.tree_root = TreeNode(state=start_state, children=[])
			ss = SearchNodeTree(state=start_state, parent=None, cost=0, tree_node=self.tree_root, \
						heuristic=self.heuristic(start_state, self.goal_state))
		else:
			ss = SearchNode(state=start_state, parent=None, cost=0, \
						heuristic=self.heuristic(start_state, self.goal_state))

		self.neighbor_count = 0

		# a global
		self.tiebreaker = count()
		heapq.heappush(self.frontier, (ss.heuristic, next(self.tiebreaker), ss))

	def search(self, time_limit):
		if self.goal_state == None: # Goal이 없는 경우 search는 수행하지 않는다
			return None
		
		start_time = time.time()
		# extend nodes until the time limit is reached
		while time.time() - start_time < time_limit: # 제한된 시간 초과 시 _step 함수 호출을 중지한다
			if len(self.frontier) == 0:
				print("Search failed, bailing early")
				return
			self._step()

	def _step(self, make_tree=False, neighbor_callback=None):
		calc_time = np.zeros(9, dtype=np.float64)
		# index
		# 0 : heappop
		# 1 : maketree
		# 2 : neighbors
		# 3 : is_admissible
		# 4 : neighbors_cbappend
		# 5 : heuristic
		# 6 : heappush
		# 7 : neighbors_cb
		# 8 : append
		""" Perform one iteration of heuristic search - extend a single node.
			Requires that self.frontier is not empty.
		"""
		# Pop the best path to split.
		t = time.time()
		parent_score, parent_count, parent_state = heapq.heappop(self.frontier)
		calc_time[0] = calc_time[0] + (time.time() - t)
		# 각각 거리값, 부모 노드 수, 현재 상태를 나타내는 것 처럼 보임
		if self.make_tree:
			t = time.time()
			parent_tree_node = parent_state.tree_node
			calc_time[1] = calc_time[1] + (time.time() - t)

		# print "Expand node:", parent_state

		if neighbor_callback:
			# neighbor_callback
			# - SE : 인접한 neighbor들의 pose와 반지름 길이를 지도에 기록하는 콜백 함수 추가?
			# - PP : SE와 동일
			neighbor_states = []

		init_neighbors = False
		init_admissible = False

		t = time.time()
		for neighbor_state in self.neighbors(parent_state.state): # 현재 위치 state를 기준으로 인접한 neighbor_state를 순회한다
			if init_neighbors is False:
				calc_time[2] = calc_time[2] + (time.time() - t)
				init_neighbors = True
			# prune any path segments that are bound to fail
			t = time.time()
			if not self.should_bail(neighbor_state, self.goal_state) and self.is_admissible(neighbor_state):
				if init_admissible is False:
					calc_time[3] = calc_time[3] + (time.time() - t)
					init_admissible = True
				# should_bail : 미구현함수 무조건 False 리턴
				# is_admissible :
				# - SpaceExploration은 neighbor_state의 반지름이 self.hard_min_radius보다 큰 경우 True 리턴, 아니면 False 리턴
				# - PathPlanner도 마찬가지임
				# 따라서 해당 if 블록은 인접한 neighbor_state의 반지름이 self.hard_min_radius보다 작아야 수행됨

				# print "   neighbor: ", neighbor_state
				# build the tree representation of the search

				heuristic_t = time.time() # Heuristic
				if self.make_tree: # True, False 일 때의 차이점은 아직 잘 모르겠음
					ntn = TreeNode(state=neighbor_state, children=[])
					parent_tree_node.children.append(ntn)
				
					nss = SearchNodeTree(state=neighbor_state, parent=parent_state, tree_node=ntn, \
						cost=parent_state.cost + self.cost(neighbor_state),  \
						heuristic=self.heuristic(neighbor_state, self.goal_state))
				else:
					nss = SearchNode(state=neighbor_state, parent=parent_state, \
						cost=parent_state.cost + self.cost(neighbor_state),  \
						heuristic=self.heuristic(neighbor_state, self.goal_state))

				if neighbor_callback:
					t = time.time()  # neighbor_cbappend
					neighbor_states.append(neighbor_state)
					calc_time[4] = calc_time[4] + (time.time() - t)

				calc_time[5] = calc_time[5] + (time.time() - heuristic_t)
				# print("\t\t\t\t{:.3f} Seconds SearchNode total".format(time.time() - t))
				self.neighbor_count += 1

				t = time.time() # Calc heappush time
				score = nss.cost + nss.heuristic # nss
				if self.goal_met(neighbor_state, self.goal_state): #
					# print "Met goal:", neighbor_state
					heapq.heappush(self.found_paths, (score, next(self.tiebreaker), nss)) # found_paths 리스트에 neighbor_state의 인스턴스 nss를 추가
				else:
					# print (score, len(self.frontier))
					heapq.heappush(self.frontier, (score, next(self.tiebreaker), nss))

				#print("heapq.heappush() terminated in:", time.time() - t, "seconds")
				calc_time[6] = calc_time[6] + (time.time() - t)


		if neighbor_callback:
			t = time.time()  # neighbor_cb
			neighbor_callback(neighbor_states)
			calc_time[7] = calc_time[7] + (time.time() - t)
		
		# self.map.add_circle_to_exploration_buffer(parent_state.state)
		t = time.time()  # Calc append
		self.closed_set.append(parent_state.state)
		calc_time[8] = calc_time[8] + (time.time() - t)
		self.step_count += 1

		return calc_time

	def best(self):
		if len(self.found_paths) > 0:
			return self.make_path(self.found_paths[0][2], True, False)
		return None   

	def make_path(self, end_node, add_goal=False, remove_start=True):
		# a path is a list of control states, in the order of traversal
		# builds it in reverse order by following parent pointers, then reverses result
		path = [end_node.state]
		while end_node.parent:
			path.append(end_node.parent.state)
			end_node = end_node.parent
		# don't execute the start state
		if remove_start:
			path.pop()
		path.reverse()

		if add_goal:
			path.append(self.goal_state)
		return Path(states=path)

	""" The following functions should be overridden by the parent class
	"""
	def cost(self, path):
		# estimate the cost of traversing a given path
		raise NotImplementedError("HeuristicSearch cost not specified")

	def heuristic(self, state, goal_state):
		# return an estimate for cost to go between the given state and the goal state
		raise NotImplementedError("HeuristicSearch heuristic not specified")

	def goal(self):
		# return the goal state
		raise NotImplementedError("HeuristicSearch goal not specified")

	def goal_met(self):
		# return true if the goal state is met
		raise NotImplementedError("HeuristicSearch goal not specified")

	def should_bail(self, state, goal_state):
		# return true if the node should be terminated
		return False

	def is_admissible(self, state):
		# return true if the given state is considered admissible - fulfills hard constraints
		raise NotImplementedError("HeuristicSearch is_admissible not specified")

	def neighbors(self, state):
		# return a discretized set of neighbors of the given state
		raise NotImplementedError("HeuristicSearch neighbors not specified")

class SpaceExploration(HeuristicSearch):
	""" This class implements lower dimensional search to provide a high quality
		heuristic for higher dimensional path planning. If this search fails to 
		provide a solution, then there is a good chance that no solution exists
		and the car should just stop

		# TODO:
			put expanded nodes into a KD tree for fast nearest neighbors search. 
				This can prevent getting stuck in dead ends

		See: 
		    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7353741
		    http://www6.in.tum.de/Main/Publications/ChenChao2013a.pdf
	"""
	def __init__(self, omap):
		# fetch relevant parameters

		# 휴리스틱 클래스 필드들..
		# self.closed_set = []
		# self.frontier = []
		# self.found_paths = []
		# self.goal_state = self.goal()
		# self.step_count = 0

		self.branch_factor    = int(rospy.get_param("~branch_factor", 31))
		self.min_turn_radius  = float(rospy.get_param("~minimum_turn_radius", 0.3))
		self.soft_min_radius  = float(rospy.get_param("~soft_min_radius", 0.8))
		#self.soft_min_radius = float(rospy.get_param("~soft_min_radius", 0.05))
		self.soft_min_penalty = float(rospy.get_param("~soft_min_penalty", 1.7))
		self.hard_min_radius  = float(rospy.get_param("~hard_min_radius", 0.3))
		#self.hard_min_radius = float(rospy.get_param("~hard_min_radius", 0.01))
		self.heuristic_bias   = float(rospy.get_param("~heuristic_bias", 1.9))
		self.min_goal_overlap = float(rospy.get_param("~min_goal_overlap", 0.05))
		self.half_space_theta = float(rospy.get_param("~half_space_theta", 0))

		# cache reused values
		self.map = omap
		self.thetas = np.zeros(self.branch_factor, dtype=float)
		self.radii  = np.zeros(self.branch_factor)
		self.euclidean_neighbors = np.zeros((self.branch_factor,2),dtype=np.float64)
		self.next_goal = None
		
		# initialize super
		super(SpaceExploration, self).__init__(make_tree=True)

	def add_states_to_exploration_buffer(self, neighbor_circles):
		if len(neighbor_circles):
			poses = np.array([c.center for c in neighbor_circles])
			radii = np.array([c.radius for c in neighbor_circles])
			self.map.add_circles_to_exploration_buffer(poses, radii)

	def reset(self, pose):
		# Frontier is a priority queue.
		start_state = Circle(center=pose[:2], radius=self.map.get_distances(np.array([pose]))[0], angle=pose[2], deflection=0)
		self.map.clear_exploration_buffer()
		super(SpaceExploration, self).reset(start_state)

	def cost(self, state):
		if state.radius < self.soft_min_radius:
			return state.radius * self.soft_min_penalty
		return state.radius

	def heuristic(self, state, goal_state):
		if goal_state == None:
			return np.inf
		return (np.linalg.norm(state.center - goal_state.center) - state.radius)*self.heuristic_bias

	def goal(self):
		return self.next_goal

	def set_goal(self, pose):
		r = self.map.get_distances(np.array([pose]))[0]
		self.next_goal = Circle(center=pose[:2], radius=r, angle=pose[2], deflection=0)

	def overlap(self, s1, s2, percentage=.15):
		# TODO/NOTE: this is a bit of a hack to go fast, percentage overlap not accurate
		if np.linalg.norm(s1.center - s2.center) > (s1.radius + s2.radius)*(1.0-percentage):
			return False
		else:
			return True

	def goal_met(self, state, goal_state):
		return self.overlap(state, goal_state, self.min_goal_overlap)

	def is_admissible(self, state):
		return state.radius > self.hard_min_radius

	def search(self, time_limit):
		# same as before, but this returns early if paths are found
		start_time = time.time()
		step_total = 0.0

		# extend nodes until the time limit is reached
		while time.time() - start_time < time_limit and not len(self.found_paths):
			if len(self.frontier) == 0:
				print("Search failed, bailing early")
				return
			t = time.time()
			self._step(neighbor_callback=self.add_states_to_exploration_buffer)
			step_total = step_total + (time.time() - t)

		print("\t\t\t{:.3f} Seconds _step() total".format(step_total))

	def neighbors(self, state):
		t = time.time()
		max_angle = utils.max_angle(self.min_turn_radius, state.radius)
		self.thetas = np.linspace(-max_angle,max_angle, num=self.branch_factor)
		self.euclidean_neighbors[:,0] = state.radius * np.cos(self.thetas + state.angle) + state.center[0]
		self.euclidean_neighbors[:,1] = state.radius * np.sin(self.thetas + state.angle) + state.center[1]

		# we use this several times, so compute this icky math here to avoid unnecessary computation
		euc = self.euclidean_neighbors.copy()
		utils.world_to_map(euc, self.map.map_info)
		euc = np.round(euc).astype(int)
		radii = self.map.get_distances(euc, coord_convert=False, check_bounds=True)
		mask = np.logical_and(np.invert(self.map.get_explored(euc, coord_convert=False)), self.map.get_permissible(euc, coord_convert=False))

		neighbors = list(map(lambda c,r,t: Circle(center=c.copy(), radius=r, angle=state.angle + t, deflection=t), 
			self.euclidean_neighbors[mask,:], radii[mask], self.thetas[mask]))
		# print("\t\t\t\t{:.3f} Seconds neighbors()".format(time.time() - t))
		return neighbors

class PathPlanner(HeuristicSearch):
	''' This class is used to refine an initial trajectory from the space
	    exploration search. It uses the original path as a heuristic and 
	    integrates a term in the cost function which penalizes excessive
	    turning. Generates suboptimal but visually smooth paths.
	'''
	def __init__(self, omap):
		self.branch_factor    = int(rospy.get_param("~fp_branch_factor", 121)) # neighbors 함수의 percentage 값에 영향을 줌
		self.min_turn_radius  = float(rospy.get_param("~fp_minimum_turn_radius", 0.4)) # neighbors 함수의 max_angle 값에 영향을 줌
		self.soft_min_radius  = float(rospy.get_param("~soft_min_radius", 0.8)) #
		self.soft_min_penalty = float(rospy.get_param("~fp_soft_min_penalty", 2.0)) #
		self.hard_min_radius  = float(rospy.get_param("~hard_min_radius", 0.3)) #
		self.heuristic_bias   = float(rospy.get_param("~fp_heuristic_bias", 1.2)) #
		self.min_goal_overlap = float(rospy.get_param("~min_goal_overlap", 0.05)) #
		self.half_space_theta = float(rospy.get_param("~half_space_theta", 0)) #
		self.exploration_coeff= float(rospy.get_param("~fp_exploration_coeff", 0.2)) #
		self.max_circle_radius= float(rospy.get_param("~fp_max_circle_radius", 1.1)) #

		self.neighbors_time = np.zeros(14, dtype=np.float64)
		self.heuristic_time = np.zeros(2, dtype=np.float64)

		self.map = omap
		self.next_goal = None
		self.rough_trajectory = None
		
		# initialize super
		super(PathPlanner, self).__init__(make_tree=True)

	def set_heursitic_trajectory(self, trajectory):
		self.rough_trajectory = trajectory

	def reset(self, pose):
		start_state = Circle(center=pose[:2], radius=self.map.get_distances(np.array([pose]))[0], angle=pose[2], deflection=0)
		self.map.clear_exploration_buffer()
		super(PathPlanner, self).reset(start_state)

	def cost(self, state):
		''' This is a fairly ad hoc cost function which tries to minimize path length and
		    curvy paths.
		'''
		other_coeff = 0.0
		if state.radius < self.soft_min_radius:
			other_coeff += 0.5
			# print "softmax"
			# return state.radius * self.soft_min_penalty + 10.0*np.power(abs(state.deflection),10.0)
		# return state.radius
		# return np.power(state.radius, 0.75) + 1.0*np.power(abs(state.deflection),3.0)
		
		# if state.deflection == 0.0:
		# 	other_coeff = -0.15
		# return state.radius + 5.0*np.power(abs(state.deflection),2.0) + other_coeff

		# if state.deflection == 0.0:
		# 	other_coeff = -0.08
		# return state.radius + 3.0*np.power(abs(state.deflection),1.2) + other_coeff

		# if state.deflection == 0.0:
		# 	other_coeff = -0.02
		return state.radius + 1.5*np.power(abs(state.deflection),1.) + other_coeff

		# if state.deflection == 0.0:
		# 	other_coeff = -0.1
		# return state.radius + 3.0*np.power(abs(state.deflection),1.) + other_coeff
		# return state.radius + 1.8*np.power(abs(state.deflection),1.0) + other_coeff

	def heuristic(self, state, goal_state):
		''' Finds the nearest point along the rough trajectory and then uses path length from there to the end as heursitic
		'''
		nearest_point, nearest_dist, t, i = utils.nearest_point_on_trajectory(state.center, self.rough_trajectory.np_points)
		distance_to_goal_along_trajectory = self.rough_trajectory.distance_to_end(i+t)
		return (distance_to_goal_along_trajectory - state.radius*2.0) * self.heuristic_bias
	
	def goal(self):
		return self.next_goal

	def set_goal(self, pose):
		r = self.map.get_distances(np.array([pose]))[0]
		r = np.clip(r, 0.0, self.max_circle_radius)
		self.next_goal = Circle(center=pose[:2], radius=r, angle=pose[2], deflection=0)

	def overlap(self, s1, s2, percentage=.15):
		# TODO/NOTE: this is a bit of a hack to go fast, percentage overlap not accurate
		if np.linalg.norm(s1.center - s2.center) > (s1.radius + s2.radius)*(1.0-percentage):
			return False
		else:
			return True

	def goal_met(self, state, goal_state): # 원 영역이 겹치는 경우 도착한걸로 판단함을 알 수 있습니다.
		return self.overlap(state, goal_state, self.min_goal_overlap)

	def is_admissible(self, state):
		return state.radius > self.hard_min_radius

	def search(self, time_limit):
		# same as before, but this returns early if paths are found
		start_time = time.time()

		calc_time = np.zeros(4, dtype=np.float64)

		step_total = 0.0
		heuristic_total = 0.0
		heappush_total = 0.0
		append_total = 0.0
		heappop_total = 0.0
		neighbors_total = 0.0
		admissible_total = 0.0
		neighbors_cbappend_total = 0.0
		neighbors_cb_total = 0.0

		# extend nodes until the time limit is reached
		# print("Len :", len(self.found_paths))

		while time.time() - start_time < time_limit and not len(self.found_paths):
			if len(self.frontier) == 0:
				print("Search failed, bailing early")
				return
			t = time.time()
			heappop, maketree, neighbors, is_admissible, neighbors_cbappend, heuristic, heappush, neighbors_cb, append = self._step(neighbor_callback=self.add_states_to_exploration_buffer)

			step_total = step_total + (time.time() - t)
			heuristic_total = heuristic_total + heuristic
			heappush_total = heappush_total + heappush
			append_total = append_total + append
			heappop_total = heappop_total + heappop
			neighbors_total = neighbors_total + neighbors
			admissible_total = admissible_total + is_admissible
			neighbors_cbappend_total = neighbors_cbappend_total + neighbors_cbappend
			neighbors_cb_total = neighbors_cb_total + neighbors_cb
		print("\t\t\t\t{:.3f} Seconds heuristic total".format(heuristic_total))
		print("\t\t\t\t{:.3f} Seconds heappush total".format(heappush_total))
		print("\t\t\t\t{:.3f} Seconds append total".format(append_total))
		print("\t\t\t\t{:.3f} Seconds heappop total".format(heappop_total))
		print("\t\t\t\t{:.3f} Seconds neighbors total".format(neighbors_total))
		print("\t\t\t\t\t{:.3f} Seconds neighbors - utils.max_angle()".format(self.neighbors_time[0]))
		print("\t\t\t\t\t{:.3f} Seconds neighbors - np.power()".format(self.neighbors_time[1]))
		print("\t\t\t\t{:.3f} Seconds is_admissible total".format(admissible_total))
		print("\t\t\t\t{:.3f} Seconds neighbors_cbappend total".format(neighbors_cbappend_total))
		print("\t\t\t\t{:.3f} Seconds neighbors_cb total".format(neighbors_cb_total))
		print("\t\t\t{:.3f} Seconds _step() total".format(step_total))

		# 시간 합 초기화 진행
		self.neighbors_time = np.zeros(14, dtype=np.float64)
		self.heuristic_time = np.zeros(2, dtype=np.float64)

	def add_states_to_exploration_buffer(self, neighbor_circles):
		if len(neighbor_circles):
			poses = np.array([c.center for c in neighbor_circles])
			radii = np.array([c.radius for c in neighbor_circles])
			self.map.add_circles_to_exploration_buffer(poses, radii, exp_coeff=self.exploration_coeff)

	def neighbors(self, state):
		''' This function determines the set of points along the perimeter of the given
		    circle state that are accessible according to ackermann geometry and steering
		    angle limitations. Then it samples the permissible theta space to generate neighbors.

		    TODO: computation of neighboring thetas is not accurate - it should actually be
		          the direction tangent to the path arc at the intersection with the circle radius.
		          For now this works, and is a strict underestimate of the reachable thetas
		'''
		t = time.time()
		max_angle = utils.max_angle(self.min_turn_radius, state.radius)
		self.neighbors_time[0] = self.neighbors_time[0] + (time.time() - t)

		t = time.time()
		percentage = np.power(2.0 * self.branch_factor / np.pi, 0.9)
		self.neighbors_time[1] = self.neighbors_time[1] + (time.time() - t)

		t = time.time()
		actual_branch_factor = 2*int(max_angle * percentage / 2.0)+1
		self.neighbors_time[2] = self.neighbors_time[2] + (time.time() - t)

		t = time.time()
		thetas = np.linspace(-max_angle,max_angle, num=actual_branch_factor) # 생성할 갈래 수?
		self.neighbors_time[3] = self.neighbors_time[3] + (time.time() - t)
		# example
		# max angle = 30
		# -30 ~ 30 사이 값들에 대한 평균분포를 리스트로 리턴해서 thetas에 저장,
		# 이 때 생성되는 각도 값들은 actual_branch_factor만큼 생성된다.

		t = time.time()
		euclidean_neighbors = np.zeros((actual_branch_factor,2))
		self.neighbors_time[4] = self.neighbors_time[4] + (time.time() - t)
		# euclidean_neighbors
		# 원 영역 내에서 이동가능한 점 리스트를 정의하기 위한 리스트
		t = time.time()
		euclidean_neighbors[:,0] = state.radius * np.cos(thetas + state.angle) + state.center[0]
		self.neighbors_time[5] = self.neighbors_time[5] + (time.time() - t)

		t = time.time()
		euclidean_neighbors[:,1] = state.radius * np.sin(thetas + state.angle) + state.center[1]
		self.neighbors_time[6] = self.neighbors_time[6] + (time.time() - t)
		# x, y 각각의 점에 대해서 계산을 수행한다.

		# perform coordinate space conversion here and then index into the exploration and permissible
		# buffers to prune states which are not permissible or already explored
		t = time.time()
		euc = euclidean_neighbors.copy()
		self.neighbors_time[7] = self.neighbors_time[7] + (time.time() - t)

		t = time.time()
		utils.world_to_map(euc, self.map.map_info) ## 부하량 2순위
		self.neighbors_time[8] = self.neighbors_time[8] + (time.time() - t)

		t = time.time()
		euc = np.round(euc).astype(int) # 넘파이 배열 euc에 저장된 값을 정수로 변환
		self.neighbors_time[9] = self.neighbors_time[9] + (time.time() - t)

		t = time.time()
		radii = self.map.get_distances(euc, coord_convert=False, check_bounds=True) - 0.05 ## 부하량 1순위
		self.neighbors_time[3] = self.neighbors_time[3] + (time.time() - t)
		# euc에 저장된 값들을 사용해서 지도 내 실제 이동 거리 계산?

		t = time.time()
		radii = np.clip(radii, 0.0, self.max_circle_radius)
		# radii에 저장된 값 중에서 0.0보다 작은 값은 0으로,
		# self.max_circle_radius 보다 큰 값은 self.max_circle_radius 로 변경한다.
		t = time.time()
		mask = np.logical_and(np.invert(self.map.get_explored(euc, coord_convert=False)), self.map.get_permissible(euc, coord_convert=False))
		# euc에 저장된 값들을 사용해서 지도 내에서 해당 위치까지 이동 가능한지 확인 진행
		# mask에 저장되는 값들은 boolean 값들이 저장됨

		# generate a neighbor state for each of the permissible
		t = time.time()
		neighbors = list(map(lambda c,r,t: Circle(center=c.copy(), radius=r, angle=state.angle + t, deflection=t), 
			euclidean_neighbors[mask,:], radii[mask], thetas[mask]))

		return neighbors

class FindTrajectory(object):
	""" Performs two levels of heuristic search to plan paths between the robot pose and a clicked goal pose.
	"""
	def __init__(self):
		self.found_trajectory  = utils.LineTrajectory("/found_trajectory")
		self.rough_trajectory  = utils.LineTrajectory("/rough_trajectory")
		self.should_publish    = bool(rospy.get_param("~publish", 1))
		self.pub_topic         = rospy.get_param("~trajectory_topic", "/trajectory/current")
		self.odom_topic        = rospy.get_param("~odom_topic", "/pf/pose/odom")
		self.exploration_timeout  = rospy.get_param("~exploration_timeout", 6.0) # rough search time limit
		self.show_exploration_buffer = rospy.get_param("~show_exploration_buffer", 1)
		self.save_trajectory   = rospy.get_param("~save_trajectory", 0)
		self.save_path         = rospy.get_param("~save_path", "/home/ros/wecar_ws/src/MIT-Racecar-Labs/trajectories/waypoint/")

		self.should_refine_trajectory = bool(rospy.get_param("~refine_trajectory", 1))
		self.refining_timeout  = rospy.get_param("~refining_timeout", 13.0) # refining trajectory time limit

		self.map = None
		self.map_initialized = False
		self.last_pose_time = 0.0
		self.circle_path = None
		
		# just a few initial poses for debugging purposes
		# self.last_pose = np.array([-1., 0., -np.pi/1.3])
		# self.last_pose = np.array([-1., 0., 0.0])
		self.last_pose = np.array([-3., -1., np.pi])
		# self.last_pose = np.array([-1.27, -2.6, np.pi])
		# self.last_pose = np.array([-6.27, -2.6, np.pi])
		# self.last_pose = np.array([15.3, 25.18, 0])
		# self.last_pose = np.array([-10,  33.8, 0])
		# self.last_pose = None

		self.get_omap()
		self.space_explorer = SpaceExploration(self.map)

		if self.should_publish:
			self.traj_pub = rospy.Publisher(self.pub_topic, PolygonStamped, queue_size=1)

		if self.should_refine_trajectory:
			self.path_planner = PathPlanner(self.map)
			self.fast_path = None
			self.fast_trajectory = utils.LineTrajectory("/fast_trajectory")

		# visualization publishers
		self.viz_namespace = "/circle_search"
		self.circle_pub = rospy.Publisher(self.viz_namespace + "/exploration_circles", MarkerArray, queue_size=1)
		self.fast_circle_pub = rospy.Publisher(self.viz_namespace + "/fast_circles", MarkerArray, queue_size=1)
		self.exploration_pub = rospy.Publisher(self.viz_namespace + "/exploration_buffer", OccupancyGrid, queue_size=1)

		self.nav_goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_point_callback, queue_size=1)
		self.odom_sum = rospy.Subscriber(self.odom_topic,  Odometry, self.odom_callback, queue_size=1)

		print("Initialized. Waiting on messages...")

	def get_omap(self):
		map_service_name = rospy.get_param("~static_map", "static_map")
		print("Fetching map from service: ", map_service_name)
		rospy.wait_for_service(map_service_name)
		map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
		self.map = utils.Map(map_msg)
		self.map_initialized = True
		print("Finished loading map")

	def publish_trajectory(self):
		print("Publishing trajectory to:", self.pub_topic)
		self.traj_pub.publish(self.rough_trajectory.toPolygon())

	def non_overlapping_paths(self, node):
		# given a tree data structure, this function will return list of lists of the
		# node "state" attributes
		# if visualized as a tree, these states will not contain overlapping segments
		if not node.children:
			return [node.state]
		else:
			paths = []

			for child in node.children:
				child_paths = self.non_overlapping_paths(child)

				if type(child_paths[0]) == list:
					# child is not a leaf node, add self to first path and store
					child_paths[0].insert(0, node.state)
					paths = paths + child_paths
				else:
					# this is a leaf node, add self to path and store
					child_paths.insert(0, node.state)
					paths.append(child_paths)

			return paths
			
	def visualize(self):
		if self.circle_pub.get_num_connections() > 0 and self.circle_path:
			# print("CIRCLE PUB")

			# root = self.space_explorer.tree_root
			# markers = [utils.marker_clear_all("/map")]
			# explored = self.non_overlapping_paths(root)
			# if type(explored[0]) == list:
			# 	explored = reduce(lambda x,y: x+y, explored)
			# print "explored: ", len(explored)
			# markers += [utils.marker_from_circle(circle, index=i, linewidth=0.01, color=ColorRGBA(0, 1, 0, 0.1), lifetime=1.0) \
			#             for i, circle in enumerate(explored)]
			
			# marker_array = MarkerArray(markers=markers)
			# self.circle_pub.publish(marker_array)
			# print len(self.circle_path.states)
			# marker = utils.marker_from_circles(self.circle_path.states, 1, "Markers_NS2", "/map", 3.0, [1.,0.,0.])
			# self.circle_pub.publish(marker)

			markers = [utils.marker_clear_all("/map")]
			markers += [utils.marker_from_circle(circle, index=i, linewidth=0.05, color=ColorRGBA(1, 0, 0, 0.4), \
						lifetime=4.0)
						for i, circle in enumerate(self.circle_path.states)]
			# print len(markers)
			marker_array = MarkerArray(markers=markers)
			self.circle_pub.publish(marker_array)

		if self.should_refine_trajectory and self.fast_circle_pub.get_num_connections() > 0 and self.fast_path:
			# print("FAST CIRCLE PUB")

			# root = self.path_planner.tree_root
			# markers = [utils.marker_clear_all("/map")]
			# explored = self.non_overlapping_paths(root)
			# if type(explored[0]) == list:
			# 	explored = reduce(lambda x,y: x+y, explored)
			# print "explored: ", len(explored)
			# markers += [utils.marker_from_circle(circle, index=i, linewidth=0.01, color=ColorRGBA(0, 0, 1, 0.1), lifetime=1.0) \
			#             for i, circle in enumerate(explored)]
			markers = [utils.marker_clear_all("/map")]
			markers += [utils.marker_from_circle(circle, index=i, linewidth=0.05, color=ColorRGBA(1, 0, 0, 0.4), \
						lifetime=4.0)
						for i, circle in enumerate(self.fast_path.states)]

			marker_array = MarkerArray(markers=markers)
			self.fast_circle_pub.publish(marker_array)

		if self.show_exploration_buffer and self.exploration_pub.get_num_connections() > 0:
			print("Publishing exploration grid")
			self.exploration_pub.publish(self.map.get_exploration_occupancy_grid())
		# plt.imshow(self.map.exploration_buffer)
		# plt.imshow(self.map.permissible_region)
		# plt.show()

	def goal_point_callback(self, msg):
		with cProfile.Profile() as pr:
			print()
			print()
			cb_t = time.time()
			''' Initialize path search. First performs and exploration search, and then a second search which uses
				the result from the first search as a heuristic. Roughly respects dynamic constraints of the car.
			'''
			t = time.time()
			self.goal = np.array([msg.pose.position.x, msg.pose.position.y, utils.quaternion_to_angle(msg.pose.orientation)])
			print("\t{:.3f} Seconds self.goal = np.array()".format(time.time() - t))
			# print("\nNew goal:", self.goal)

			# print("pr.dump_stats('home/ros/pp_log.stats')")
			# pr.dump_stats("home/ros/pp_log.stats")
			# print("pr.print_stats()")
			# pr.sort_stats(SortKey.TIME).print_stats(30)
			# stats = pstats.Stats(pr).sort_stats('cumtime')
			# pr.print_stats()
			# stats.pr.print_stats()

		print("{:.3f} Seconds goal_point_callback terminated in:".format(time.time() - cb_t))

	def find_rough_trajectory(self):
		# print("Finding rough trajectory")
		# rough_t = time.time()
		t = time.time()
		self.space_explorer.set_goal(self.goal)
		print("\t\t{:.3f} Seconds space_explorer.set_goal(self.goal)".format(time.time() - t))

		t = time.time()
		self.space_explorer.reset(self.last_pose.copy())
		print("\t\t{:.3f} Seconds space_explorer.reset(self.last_pose.copy())".format(time.time() - t))

		t = time.time()
		self.space_explorer.search(self.exploration_timeout)
		print("\t\t{:.3f} Seconds space_explorer.search(self.exploration_timeout)".format(time.time() - t))
		# print("... rough search terminated in:", time.time() - t, "seconds")
		t = time.time()
		# print("... expanded", self.space_explorer.step_count, "in:", time.time() - t, "seconds")
		t = time.time()
		# print("... considered", self.space_explorer.neighbor_count, "in:", time.time() - t, "seconds")
		self.circle_path = self.space_explorer.best()
		if self.circle_path == None:
			print("...search failed")
		# print("\t\t{:.3f} Seconds find_rough_trajectory()".format(time.time() - rough_t))

	def refine_trajectory(self):
		# print("Refining trajectory")
		# refine_t = time.time()
		t = time.time()
		self.rough_trajectory.make_np_array()
		print("\t\t{:.3f} Seconds rough_trajectory.make_np_array()".format(time.time() - t))

		t = time.time()
		self.path_planner.set_heursitic_trajectory(self.rough_trajectory)
		print("\t\t{:.3f} Seconds path_planner.set_heursitic_trajectory(self.rough_trajectory)".format(time.time() - t))

		t = time.time()
		self.path_planner.set_goal(self.goal)
		print("\t\t{:.3f} Seconds path_planner.set_goal(self.goal)".format(time.time() - t))

		t = time.time()
		self.path_planner.reset(self.last_pose.copy())
		print("\t\t{:.3f} Seconds path_planner.reset(self.last_pose.copy())".format(time.time() - t))

		t = time.time()
		self.path_planner.search(self.refining_timeout)
		print("\t\t{:.3f} Seconds path_planner.search(self.refining_timeout)".format(time.time() - t))
		# t = time.time()
		# print("... expanded", self.path_planner.step_count, "in:", time.time() - t, "seconds")
		t = time.time()
		# print("... considered", self.path_planner.neighbor_count, "in:", time.time() - t, "seconds")
		self.fast_path = self.path_planner.best()
		if self.fast_path == None:
			print("... search failed")

		# print("\t\t{:.3f} Seconds refine_trajectory()".format(time.time() - refine_t))

	def has_recent_pose(self):
		# return True # this is useful for debugging
		return time.time() - self.last_pose_time < 0.5

	def odom_callback(self, msg):
		if not self.has_recent_pose():
			print("Recieved Odometry...")
		self.last_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, utils.quaternion_to_angle(msg.pose.pose.orientation)])
		self.last_pose_time = time.time()

class DefinedWaypoints():
	def __init__(self):
		self.pub = rospy.Publisher("/defined_waypoints", MarkerArray, queue_size=1)
		self.waypointlist = MarkerArray()

		# 해당 좌표 리스트들은 향후 특정 파일에서 불러오도록 수정해야 함
		self.pointlist = [
			[3.764, 0.046, 0.0], # 0
			[-37.097, 0.373, 0.0], # 1
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

	def add_waypoint(self, idx, point):
		waypoint = Marker()

		waypoint.id = idx  # Marker id value, no duplicates
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

	def publish(self):
		print("publish")
		self.pub.publish(self.waypointlist)

def make_flamegraph(filterx=None):
    import flamegraph
    perf_log_path = os.path.join(os.path.dirname(__file__), "../tmp/perf2.log")
    flamegraph.start_profile_thread(fd=open(perf_log_path, "w"),
                                    filter=filterx,
                                    interval=0.001)

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
