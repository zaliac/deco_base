import os
from os.path import join

DIST_MATRIX_PATH = 'data/smpl/smpl_neutral_geodesic_dist.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl/'
SMPLX_MODEL_DIR = 'data/smplx/'

N_PARTS = 24

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'datasets/Release_Datasets'
CONTACT_MAPPING_PATH = 'data/conversions'

# ``data/preprocess/rich_smplx.py`` relinks DECO's legacy RICH crop paths to
# the public RICH image layout.  Prefer that local archive when it has been
# prepared, while retaining the released archive as a clone-safe fallback.
_RICH_PREPARED_TRAIN_NPZ = join(
    DATASET_NPZ_PATH, 'rich/rich_train_smplx_cropped_bmp_public.npz'
)
RICH_TRAIN_NPZ = os.environ.get(
    'DECO_RICH_TRAIN_NPZ',
    _RICH_PREPARED_TRAIN_NPZ
    if os.path.isfile(_RICH_PREPARED_TRAIN_NPZ)
    else join(DATASET_NPZ_PATH, 'rich/rich_train_smplx_cropped_bmp.npz'),
)
_RICH_PREPARED_TEST_NPZ = join(
    DATASET_NPZ_PATH, 'rich/rich_test_smplx_cropped_bmp_public.npz'
)
RICH_TEST_NPZ = os.environ.get(
    'DECO_RICH_TEST_NPZ',
    _RICH_PREPARED_TEST_NPZ
    if os.path.isfile(_RICH_PREPARED_TEST_NPZ)
    else join(DATASET_NPZ_PATH, 'rich/rich_test_smplx_cropped_bmp.npz'),
)

# Path to test/train npz files
DATASET_FILES = {
    'train': {
        'damon': join(DATASET_NPZ_PATH, 'damon/hot_dca_trainval_with_kpts.npz'),
        'rich': RICH_TRAIN_NPZ,
        'prox': join(DATASET_NPZ_PATH, 'prox/prox_train_smplx_ds4.npz'),
    },
    'val': {
        'damon': join(DATASET_NPZ_PATH, 'damon/hot_dca_test_with_kpts.npz'),
        'rich': RICH_TEST_NPZ,
        'prox': join(DATASET_NPZ_PATH, 'prox/prox_val_smplx_ds4.npz'),
        'behave': join(DATASET_NPZ_PATH, 'behave/behave_test.npz'),
    },
    'test': {
        'damon': join(DATASET_NPZ_PATH, 'damon/hot_dca_test_with_kpts.npz'),
        'rich': RICH_TEST_NPZ,
        'prox': join(DATASET_NPZ_PATH, 'prox/prox_val_smplx_ds4.npz'),
        'behave': join(DATASET_NPZ_PATH, 'behave/behave_test.npz'),
    },
}
