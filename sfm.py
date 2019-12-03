import cv2
import numpy as np

K = np.load('OP5_camera_matrix.npy')


class SFM:

    def __init__(self, views):
        super(SFM, self).__init__()

        self.views = views
        self.points_3D = np.zeros((0, 3), dtype=np.float32)
        self.num_points_3D = 0
        self.done = []

    def baseline_pose(self, view1, view2):

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
        matched_pixels1 = np.array(indices1)
        matched_pixels2 = np.array(indices2)

        F, mask = cv2.findFundamentalMat(matched_pixels1, matched_pixels2, method=cv2.FM_RANSAC,
                                         ransacReprojThreshold=0.9, confidence=0.9)
        mask = mask.astype(bool).flatten()

        E = K.transpose().dot(F).dot(K)

        matched_pixels1 = matched_pixels1[mask]
        indices1 = indices1[mask]
        matched_pixels2 = matched_pixels2[mask]
        indices2 = indices2[mask]

        _, R, t, _ = cv2.recoverPose(E, matched_pixels1, matched_pixels2, K)

        view1.rotation = np.eye(3, 3)
        view1.translation = np.zeros(3, 1)

        view2.rotation = R
        view2.translation = t

        point_indices = np.arange(self.num_points_3D, len(indices1))
        view1.indices[indices1] = point_indices
        view2.indices[indices2] = point_indices

        return matched_pixels1, matched_pixels2

    def triangulate_points(self, view1, view2, matched_pixels1, matched_pixels2):

        points1_homogenous = cv2.convertPointsToHomogeneous(matched_pixels1)[:, 0, :]
        points2_homogenous = cv2.convertPointsToHomogeneous(matched_pixels2)[:, 0, :]

        points1_normalized = np.linalg.inv(K).dot(points1_homogenous.transpose()).transpose()
        points2_normalized = np.linalg.inv(K).dot(points2_homogenous.transpose()).transpose()

        triangulated_points_homogenous = cv2.triangulatePoints(np.hstack((view1.rotation, view1.translation)),
                                                               np.hstack((view2.rotation, view2.translation)),
                                                               points1_normalized.transpose(), points2_normalized.transpose())
        triangulated_points = cv2.convertPointsFromHomogeneous(triangulated_points_homogenous.transpose)[:, 0, :]

        self.points_3D = np.concatenate((self.points_3D, triangulated_points))
        self.num_points_3D += self.points_3D.shape[0]

    def save_ply(self, filename):

        colors = np.zeros(self.points_3D.shape, dtype=np.int32)

        for view in self.done:

            kps = np.array(view.keypoints)[view.indices > 0]
            img_pts = [kp.pt for kp in kps]
            img = cv2.imread('images/' + view.name)

            colors[view.indices[view.indices > 0].astype(int)] = img[np.int32(img_pts[:,1]),
                                                                     np.int32(img_pts[:,0])]

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

        img_pts1, img_pts2 = self.baseline_pose(view1=view0, view2=view1)

        self.triangulate_points(view0, view1, img_pts1, img_pts2)

        self.save_ply('./results/)_1.ply')

        self.done.append(view0)
        self.done.append(view1)









