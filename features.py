import numpy as np
import argparse
import cv2
import pdb
import os
import sys
import pickle

class View():

	def __init__(self, image_path, root_path, feat_det_type= 'sift'):
		super(View, self).__init__()

		count_parent_dir = len(os.path.dirname(image_path)) + 1
		self.file_name = image_path[count_parent_dir:-4]
		self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		self.keypoints = []
		self.descriptors = []
		self.method = feat_det_type
		self.rotation_matrix = np.zeros((3, 3), dtype=np.float32)
		self.translation_vector = np.zeros((3, 1), dtype=np.float32)
		self.matches = []
		self.indices = []
		self.root_path = root_path

	def scale_image_height(self, img, new_height):

		height, width = img.shape
		ratio = width/height
		new_width = int(new_height * ratio)
		dim = (new_width, new_height)

		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

		return resized

	def write_features_file(self):

		if not os.path.exists(self.root_path + '/features'):
			os.makedirs(self.root_path + '/features')

		temp_array = []
		for idx, point in enumerate(self.keypoints):
			temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, self.descriptors[idx])
			temp_array.append(temp)
		
		features_file = open(self.root_path + '/features/'+ self.file_name + '.pkl', 'wb')
		pickle.dump(temp_array, features_file) 
		features_file.close()

	def read_features_file(self):

		features = pickle.load( open(self.root_path + '/features/'+ self.file_name + '.pkl', "rb" ) )

		keypoints = []
		descriptors = []

		for point in features:
			temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
			temp_descriptor = point[6]
			keypoints.append(temp_feature)
			descriptors.append(temp_descriptor)
		return keypoints, np.array(descriptors)

	def write_matches_file(self, previous_view_name):

		if not os.path.exists(self.root_path + '/matches'):
			os.makedirs(self.root_path + '/matches')

		temp_array = []
		for idx, match in enumerate(self.matches):
			temp = (match.distance, match.imgIdx, match.queryIdx, match.trainIdx)
			temp_array.append(temp)
		
		matches_file = open(self.root_path + '/matches/'+ previous_view_name + '_' + self.file_name + '.pkl', 'wb')
		pickle.dump(temp_array, matches_file) 
		matches_file.close()

	def read_matches_file(self, filename):

		matches = []

		file_matches = pickle.load( open(filename, "rb" ) )

		for point in file_matches:
			temp_feature = cv2.DMatch(_distance=point[0], _imgIdx=point[1], _queryIdx=point[2], _trainIdx=point[3])
			matches.append(temp_feature)

		self.matches = matches

	def extract_features(self, features_path):

		if features_path != None:
			self.keypoints, self.descriptors = self.read_features_file()

		else:

			if self.method == 'sift':
				detector = cv2.xfeatures2d.SIFT_create()
			elif self.method == 'surf':
				detector = cv2.xfeatures2d.SURF_create()
			elif self.method == 'orb':
				detector = cv2.ORB_create(nfeatures=1500) # Without nfeatures value it detects fewer than SIFT and SUFT
			else:
				print('Admitted values for the feature detector are: sift, surf or orb ')
				sys.exit(0)

			self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)

			self.write_features_file()

		self.indices = np.full(len(self.keypoints), -1)

def get_files_paths(folder_path):


	files_paths = []

	if folder_path != None:
		# r=root, d=directories, f = files
		for r, d, f in os.walk(folder_path):
			for file in f:
				if '.jpg' in file or '.pkl':
					files_paths.append(os.path.join(r, file))

	if len(files_paths) == 0:
		return None

	else:
		return files_paths

def match(descriptors_1, descriptors_2, feature_detection, matcher_alg='brute_force', distance_type=''):

	# Brute Force Matching

	# Params: First one is normType. It specifies the distance measurement to be used. By default, it is cv.NORM_L2. It is good for SIFT, SURF etc 
	# (cv.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used
	# Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
	# Second param is boolean variable, crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) 
	# such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets 
	# should match each other.

	closest_matches = []
	ransac_matches = []
	crossCheck = False if distance_type == 'ratio' else True

	if feature_detection == 'sift' or feature_detection == 'surf':

		if matcher_alg == 'brute_force':
			matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)

		elif matcher_alg == 'flann': 
			FLANN_INDEX_KDTREE = 1
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks=50)   # or pass empty dictionary

			matcher = cv2.FlannBasedMatcher(index_params,search_params)
		
	elif feature_detection == 'orb':

		if matcher_alg == 'brute_force':
			matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck) 

		elif matcher_alg == 'flann': 
			FLANN_INDEX_LSH = 6
			index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
			search_params = dict(checks=50)   # or pass empty dictionary

			matcher = cv2.FlannBasedMatcher(index_params,search_params)

	if distance_type == 'ratio':

		matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)

		# ratio test
		for m,n in matches:
			if m.distance < 0.7*n.distance:

				closest_matches.append(m)
	else:

		matches = matcher.match(descriptors_1, descriptors_2) # match() returns the best match
		matches = sorted(matches, key = lambda x: x.distance)
		#matches = matches[:50]

		for m in matches:
			if m.distance < 170: #0.5*matches[-1].distance: 
				closest_matches.append(m)

	return closest_matches

def match_views(view1, view2, matcher_alg='brute_force', distance_type=''):

	matches_paths = get_files_paths(view2.root_path + '/matches/')

	filename = view2.root_path + '/matches/'+ view1.file_name + '_' + view2.file_name + '.pkl'

	if matches_paths != None and os.path.isfile(filename):

		view2.read_matches_file(filename)

	else:

		ransac_matches = match(view1.descriptors, view2.descriptors, view2.method, matcher_alg, distance_type)

		view2.matches = ransac_matches

		view2.write_matches_file(view1.file_name)

	result = cv2.drawMatches(view1.image, view1.keypoints, view2.image, view2.keypoints, view2.matches, None, flags = 2)

	return result

def create_views(root_path):

	images_paths = get_files_paths(root_path + '/images')
	features_paths = get_files_paths(root_path + '/features')

	images_paths.sort()
	if features_paths != None: features_paths.sort()

	views = []

	for i in range(len(images_paths)):

		view = View(images_paths[i], root_path, 'sift')
		feat_path = None if features_paths == None else features_paths[i]
		view.extract_features(feat_path)
		views.append(view)

	return views

