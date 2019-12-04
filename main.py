from sfm import SFM
from features import *

if __name__ == '__main__':

    views = crete_views('./benchmark')

    for i in range(1, len(views)):
        _ = match_views(views[i - 1], views[i])

    K = np.load('benchmark_intrinsic_matrix.npy')

    sfm = SFM(views=views, K=K, image_folder='benchmark')
    sfm.reconstruct()
