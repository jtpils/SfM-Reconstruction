import numpy as np
import argparse
import cv2
import pdb
import os
import sys


class View():

	def __init__(self, image_path, feat_det_type= 'sift'):
		super(View, self).__init__()

		self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		self.keypoints = []
		self.descriptors = []
		self.method = feat_det_type
		self.projection_matrix = np.zeros((3, 4), dtype=np.float32)
		self.matched_pixels = []

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

	def extract_features(self):

		if self.method == 'sift':
			detector = cv2.xfeatures2d.SIFT_create()
		elif self.method == 'surf':
			detector = cv2.xfeatures2d.SURF_create()
		elif self.method == 'orb':
			detector = cv2.ORB_create(nfeatures=1500) # Without nfeatures value it detects fewer than SIFT and SUFT
		else:
			print('Admitted values for the feature detector are: sift, surf or orb ')
			sys.exit(0)

		# Resizing step
		# pdb.set_trace()
		#self.image_1 = self.scale_image_percentage(self.image_1)
		#image_1_height = self.image_1.shape[0]
		standard_height = 1024
		self.image = self.scale_image_height(self.image, standard_height)

		self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)
		#keypoints_2, img_descriptors_2 = detector.detectAndCompute(self.image_2, None)

def get_images_paths(folder_path):
	images_path = []

	# r=root, d=directories, f = files
	for r, d, f in os.walk(training_path):
		for file in f:
			if '.jpg' in file:
				images_path.append(os.path.join(r, file))
	return images_path

# returns two values: result to plot matches, list with coordinates of matches
def match(view1, view2, matcher_alg='brute_force', distance_type=''):

	# Brute Force Matching

	# Params: First one is normType. It specifies the distance measurement to be used. By default, it is cv.NORM_L2. It is good for SIFT, SURF etc 
	# (cv.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used
	# Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
	# Second param is boolean variable, crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) 
	# such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets 
	# should match each other.

	matched_points = []
	closest_matches = []
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
		matches_mask = [[0,0] for i in range(len(matches))]

		# ratio test
		#for i,(m,n) in enumerate(matches):
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				#matches_mask[i]=[1,0]
				closest_matches.append([m])
				matched_points.append([view1.keypoints[m.queryIdx].pt, view2.keypoints[m.trainIdx].pt])

		draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matches_mask, flags = 0)
		#result = cv2.drawMatchesKnn(view1.image, view1.keypoints, view2.image, view2.keypoints, matches, None, **draw_params)
		result = cv2.drawMatchesKnn(view1.image, view1.keypoints, view2.image, view2.keypoints, closest_matches, None, flags = 2)

	else:

		matches = matcher.match(view1.descriptors, view2.descriptors) # match() returns the best match
		matches = sorted(matches, key = lambda x: x.distance)
		matches = matches[:50]

		for m in matches:
			if m.distance < 170: #0.5*matches[-1].distance: 
				closest_matches.append(m)
				matched_points.append([view1.keypoints[m.queryIdx].pt, view2.keypoints[m.trainIdx].pt])

		result = cv2.drawMatches(view1.image, view1.keypoints, view2.image, view2.keypoints, closest_matches, None, flags=2) # flags=2 hydes the features that are not matched

	matched_points = np.array(matched_points)
	view2.matched_pixels = matched_points[:,1]

	return result, matched_points

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--images_path", required=False, help="path to input dataset of training images")
ap.add_argument("-i", "--image", required=True, help="path to input dataset of training images")
ap.add_argument("-j", "--image_2", required=True, help="path to input dataset of training images")
ap.add_argument("-d", "--feature_detection", default='sift', help="feature detection method. Admitted values: sift, surf, orb. Defect value: sift")

args = vars(ap.parse_args())

folder_path = args["images_path"]
image_path = args["image"]
image_path_2 = args["image_2"]
feature_detection = args["feature_detection"]

distance_type = 'ratio'

#images_path = get_images_paths(folder_path)


view1 = View(image_path, 'sift')
view2 = View(image_path_2, 'sift')

view1.extract_features()
view2.extract_features()

display_result, matches = match(view1, view2, 'brute_force', distance_type)

#cv2.imshow("Image_{0}".format(detector), img)
cv2.imshow("Matched result_{0}".format(feature_detection), display_result)
cv2.waitKey(0)
#cv2.imwrite('features.png', img)
cv2.destroyAllWindows()
