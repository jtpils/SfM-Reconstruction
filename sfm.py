import cv2
import numpy as np
from features import *


class SFM:

    def __init__(self, views, K, image_folder):
        super(SFM, self).__init__()

        self.views = views
        self.points_3D = np.zeros((0, 3), dtype=np.float32)
        self.num_points_3D = 0
        self.done = []
        self.K = K
        self.image_folder = image_folder

    @staticmethod
    def retrieve_points(view1, view2):

        indices1 = []
        indices2 = []
        matched_pixels1 = []
        matched_pixels2 = []

        for m in view2.matches:
            indices1.append(m.queryIdx)
            indices2.append(m.trainIdx)
            matched_pixels1.append(view1.keypoints[m.queryIdx].pt)
            matched_pixels2.append(view2.keypoints[m.trainIdx].pt)

        indices1 = np.array(indices1)
        indices2 = np.array(indices2)
        matched_pixels1 = np.array(matched_pixels1)
        matched_pixels2 = np.array(matched_pixels2)

        return indices1, indices2, matched_pixels1, matched_pixels2

    @staticmethod
    def filter_matches(indices1, indices2, matched_pixels1, matched_pixels2):

        F, mask = cv2.findFundamentalMat(matched_pixels1, matched_pixels2, method=cv2.FM_RANSAC,
                                         ransacReprojThreshold=0.9, confidence=0.9)
        mask = mask.astype(bool).flatten()

        matched_pixels1 = matched_pixels1[mask]
        indices1 = indices1[mask]
        matched_pixels2 = matched_pixels2[mask]
        indices2 = indices2[mask]

        return F, mask, indices1, indices2, matched_pixels1, matched_pixels2

    def baseline_pose(self, view1, view2):

        indices1, indices2, matched_pixels1, matched_pixels2 = self.retrieve_points(view1, view2)
        F, mask, indices1, indices2, matched_pixels1, matched_pixels2 = self.filter_matches(indices1,
                                                                                            indices2,
                                                                                            matched_pixels1,
                                                                                            matched_pixels2)

        E = self.K.transpose().dot(F).dot(self.K)

        _, R, t, _ = cv2.recoverPose(E, matched_pixels1, matched_pixels2, self.K)

        view1.rotation_matrix = np.eye(3, 3, dtype=np.float32)
        view1.translation_vector = np.zeros((3, 1), dtype=np.float32)

        view2.rotation_matrix = R
        view2.translation_vector = t

        point_indices = np.arange(self.num_points_3D, self.num_points_3D + len(indices1))
        view1.indices[indices1] = point_indices
        view2.indices[indices2] = point_indices

        print('Triangulating points . . . . .')
        self.triangulate_points(view1, view2, matched_pixels1, matched_pixels2)

    def triangulate_points(self, view1, view2, matched_pixels1, matched_pixels2):

        points1_homogenous = cv2.convertPointsToHomogeneous(matched_pixels1)[:, 0, :]
        points2_homogenous = cv2.convertPointsToHomogeneous(matched_pixels2)[:, 0, :]

        points1_normalized = np.linalg.inv(self.K).dot(points1_homogenous.transpose()).transpose()
        points2_normalized = np.linalg.inv(self.K).dot(points2_homogenous.transpose()).transpose()

        points1_normalized = cv2.convertPointsFromHomogeneous(points1_normalized)[:, 0, :]
        points2_normalized = cv2.convertPointsFromHomogeneous(points2_normalized)[:, 0, :]

        triangulated_points_homogenous = cv2.triangulatePoints(np.hstack((view1.rotation_matrix, view1.translation_vector)),
                                                               np.hstack((view2.rotation_matrix, view2.translation_vector)),
                                                               points1_normalized.transpose(), points2_normalized.transpose())
        triangulated_points = cv2.convertPointsFromHomogeneous(triangulated_points_homogenous.transpose())[:, 0, :]

        self.points_3D = np.concatenate((self.points_3D, triangulated_points))
        self.num_points_3D = self.points_3D.shape[0]
        print('Done - Added to point cloud . . . . .')

    def integrate_new_view(self, new_view):

        new_points_indices = []
        new_points_3D = np.zeros((0, 3), dtype=np.float32)
        new_points_2D = np.zeros((0, 2), dtype=np.float32)

        for view in self.done:
            matches = match(view.descriptors, new_view.descriptors, view.method)

            for m in matches:
                point_3d_idx = view.indices[m.queryIdx]

                if point_3d_idx > 0 and point_3d_idx not in new_points_indices:
                    new_points_indices.append(point_3d_idx)
                    point_3D = self.points_3D[point_3d_idx, :]
                    point_3D = np.expand_dims(point_3D, axis=0)
                    new_points_3D = np.concatenate((new_points_3D, point_3D))
                    point_2D = np.array(new_view.keypoints[m.trainIdx].pt)
                    point_2D = np.expand_dims(point_2D, axis=0)
                    new_points_2D = np.concatenate((new_points_2D, point_2D))

        print('Calculating pose for new view . . . . .')
        _, R, t, _ = cv2.solvePnPRansac(new_points_3D, new_points_2D, self.K, distCoeffs=None, reprojectionError=8.,
                                        confidence=0.99, flags=cv2.SOLVEPNP_DLS)
        R, _ = cv2.Rodrigues(R)
        new_view.rotation_matrix = R
        new_view.translation_vector = t

        prev_view = self.done[-1]

        indices1, indices2, matched_pixels1, matched_pixels2 = self.retrieve_points(prev_view, new_view)
        F, mask, indices1, indices2, matched_pixels1, matched_pixels2 = self.filter_matches(indices1,
                                                                                            indices2,
                                                                                            matched_pixels1,
                                                                                            matched_pixels2)

        point_indices = np.arange(self.num_points_3D, self.num_points_3D + len(indices1))
        prev_view.indices[indices1] = point_indices
        new_view.indices[indices2] = point_indices

        print('Triangulating new points . . . . .')
        self.triangulate_points(prev_view, new_view, matched_pixels1, matched_pixels2)

    def save_ply(self, filename):

        colors = np.zeros(self.points_3D.shape, dtype=np.int32)

        for view in self.done:

            kps = np.array(view.keypoints)[view.indices > 0]
            img_pts = np.array([kp.pt for kp in kps])
            img = cv2.imread(self.image_folder + '/images/' + view.file_name + '.jpg')

            colors[view.indices[view.indices > 0]] = img[np.int32(img_pts[:, 1]),
                                                         np.int32(img_pts[:, 0])]

        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(self.points_3D.shape[0]))

            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')

            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')

            f.write('end_header\n')

            colors = np.int32(colors)

            for pt, col in zip(self.points_3D, colors):
                f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2], col[0], col[1], col[2]))

    def calculate_error(self):

        pass

    def reconstruct(self):

        view0 = self.views[0]
        view1 = self.views[1]

        print('\n\nSolving baseline pose for the first two views . . . . .')
        self.baseline_pose(view1=view0, view2=view1)

        self.done.append(view0)
        self.done.append(view1)

        print('Saving point cloud to file . . . . .')
        if not os.path.exists(self.image_folder + '/results'):
            os.makedirs(self.image_folder + '/results')
        self.save_ply(self.image_folder + '/results/0_1.ply')
        print('Done - File available at ' + self.image_folder + '/results/0_1.ply . . . . .')

        for i in range(2, len(self.views)):

            print('\n\nIntegrating new view . . . . .')
            self.integrate_new_view(self.views[i])
            ply_filename = self.image_folder + '/results/' + str(i-1) + '_' + str(i) + '.ply'
            self.done.append(self.views[i])
            print('Saving point cloud to file . . . . .')
            self.save_ply(ply_filename)
            print('Done - File available at ' + ply_filename + ' . . . . .')







