import numpy as np
import argparse
import cv2
import pdb
import os
import sys
import pickle

#pySDL
#pixelShuffle() in Pytorch to upsample pixales to generate a better quality image.

#Relevance
#Future work

class View():

	def __init__(self, image_path, feat_det_type= 'sift'):
		super(View, self).__init__()

		count_parent_dir = len(os.path.dirname(image_path)) + 1
		self.file_name = image_path[count_parent_dir:-4]
		self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		self.keypoints = []
		self.descriptors = []
		self.method = feat_det_type
		self.rotation_matrix = np.zeros((3, 3), dtype=np.float32)
		self.translation_vector = np.zeros((3, 1), dtype=np.float32)
		self.matched_points = []
		self.matches = []
		self.indices = []

	def scale_image_percentage(self, img):

		scale_percent = 60 # percent of original size
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

		return resized

	def scale_image_height(self, img, new_height):

		height, width = img.shape
		ratio = width/height
		new_width = int(new_height * ratio)
		dim = (new_width, new_height)

		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

		return resized

	def write_features_file(self):

		#i = 0
		temp_array = []
		for idx, point in enumerate(self.keypoints):
			temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, self.descriptors[idx])
			#++i
			temp_array.append(temp)
		
		features_file = open('features/'+ self.file_name + '.pkl', 'wb')
		pickle.dump(temp_array, features_file) 
		features_file.close()

		# open a file, where you ant to store the data
		#kp_file = open('keypoints/'+ self.file_name + '.pkl', 'wb')
		#desc_file = open('descriptors/'+ self.file_name + '.pkl', 'wb')

		# dump information to that file
		#pickle.dump(self.keypoints, kp_file)
		#pickle.dump(self.descriptors, desc_file)

		# close the file
		#kp_file.close()
		#desc_file.close()

	def read_features_file(self):

		features = pickle.load( open('features/'+ self.file_name + '.pkl', "rb" ) )

		keypoints = []
		descriptors = []

		for point in features:
			temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
			temp_descriptor = point[6]
			keypoints.append(temp_feature)
			descriptors.append(temp_descriptor)
		return keypoints, np.array(descriptors)

	def write_matches_file(self, previous_view_name):

		temp_array = []
		for idx, match in enumerate(self.matches):
			temp = (match.distance, match.imgIdx, match.queryIdx, match.trainIdx)
			temp_array.append(temp)
		
		matches_file = open('matches/'+ previous_view_name + '_' + self.file_name + '.pkl', 'wb')
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

		# Resizing step
		#self.image_1 = self.scale_image_percentage(self.image_1)
		#image_1_height = self.image_1.shape[0]
		standard_height = 1024
		self.image = self.scale_image_height(self.image, standard_height)

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
			#keypoints_2, img_descriptors_2 = detector.detectAndCompute(self.image_2, None)

			self.write_features_file()


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


# returns two values: result to plot matches, list with coordinates of matches
def match(view1, view2, matcher_alg='brute_force', distance_type=''):

	matches_paths = get_files_paths('matches/')

	filename = 'matches/'+ view1.file_name + '_' + view2.file_name + '.pkl'

	if matches_paths != None and os.path.isfile(filename):

		view2.read_matches_file(filename)

	else:

		# Brute Force Matching

		# Params: First one is normType. It specifies the distance measurement to be used. By default, it is cv.NORM_L2. It is good for SIFT, SURF etc 
		# (cv.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used
		# Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
		# Second param is boolean variable, crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) 
		# such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets 
		# should match each other.

		#matched_points = []
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

			matches = matcher.knnMatch(view1.descriptors, view2.descriptors, k=2)

			# Need to draw only good matches, so create a mask
			#matches_mask = [[0,0] for i in range(len(matches))]

			# ratio test
			#for i,(m,n) in enumerate(matches):
			for m,n in matches:
				if m.distance < 0.7*n.distance:
					#matches_mask[i]=[1,0]
					#closest_matches.append([m])
					closest_matches.append(m)
					#matched_points.append([view1.keypoints[m.queryIdx].pt, view2.keypoints[m.trainIdx].pt])

			#result = cv2.drawMatchesKnn(view1.image, view1.keypoints, view2.image, view2.keypoints, closest_matches, None, **draw_params)
			#result = cv2.drawMatchesKnn(view1.image, view1.keypoints, view2.image, view2.keypoints, closest_matches, None, flags = 2)

		else:

			matches = matcher.match(view1.descriptors, view2.descriptors) # match() returns the best match
			matches = sorted(matches, key = lambda x: x.distance)
			#matches = matches[:50]

			for m in matches:
				if m.distance < 170: #0.5*matches[-1].distance: 
					closest_matches.append(m)
					#matched_points.append([view1.keypoints[m.queryIdx].pt, view2.keypoints[m.trainIdx].pt])

			#result = cv2.drawMatches(view1.image, view1.keypoints, view2.image, view2.keypoints, closest_matches, None, flags=2) # flags=2 hydes the features that are not matched

		src_pts = np.float32([ view1.keypoints[m.queryIdx].pt for m in closest_matches ])
		dst_pts = np.float32([ view2.keypoints[m.trainIdx].pt for m in closest_matches ])
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matches_mask = mask.ravel().tolist()

		for idx, m in enumerate(closest_matches):
			if mask[idx] != 0:
				ransac_matches.append(closest_matches[idx])

		view2.matches = ransac_matches

		view2.write_matches_file(view1.file_name)

		#matched_points = np.array(matched_points)
		#view2.matched_points = matched_points[:,1]

	#draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matches_mask, flags = 2)
	#result = cv2.drawMatches(view1.image, view1.keypoints, view2.image, view2.keypoints, closest_matches, None, **draw_params)
	result = cv2.drawMatches(view1.image, view1.keypoints, view2.image, view2.keypoints, view2.matches, None, flags = 2)

	return result #, matched_points

def crete_views(images_path, features_paths):

	views = []

	for i in range(len(images_path)):

		view = View(images_path[i], 'sift')
		feat_path = None if features_paths == None else features_paths[i]
		view.extract_features(feat_path)
		views.append(view)

	return views

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_path", required=True, help="path to input dataset of training images")
ap.add_argument("-f", "--features_path", required=False, help="path to computed features")
#ap.add_argument("-i", "--image", required=True, help="path to input dataset of training images")
#ap.add_argument("-j", "--image_2", required=True, help="path to input dataset of training images")
ap.add_argument("-d", "--feature_detection", default='sift', help="feature detection method. Admitted values: sift, surf, orb. Defect value: sift")

args = vars(ap.parse_args())

images_path = args["images_path"]
features_path = args["features_path"]
#image_path = args["image"]
#image_path_2 = args["image_2"]
feature_detection = args["feature_detection"]

distance_type = 'ratio'

images_paths = get_files_paths(images_path)
features_paths = get_files_paths(features_path)

views = crete_views(images_paths, features_paths)

view1 = views[1]
view2 = views[2]

display_result = match(view1, view2, 'brute_force', distance_type)

#cv2.imshow("Image_{0}".format(detector), img)
cv2.imshow("Matched result_{0}".format(feature_detection), display_result)
cv2.waitKey(0)
#cv2.imwrite('features.png', img)
cv2.destroyAllWindows()
