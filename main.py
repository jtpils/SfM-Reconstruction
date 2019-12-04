from sfm import SFM
from features import *

if __name__ == '__main__':

    image_paths = get_files_paths('./benchmark')
    image_features = get_files_paths('./features')

    image_paths.sort()
    image_features.sort()

    views = crete_views(image_paths, image_features)

    for i in range(1, len(views)):
        _ = match_views(views[i - 1], views[i])

    K = np.load('benchmark_intrinsic_matrix.npy')

    sfm = SFM(views=views, K=K, image_folder='benchmark')
    sfm.reconstruct()
