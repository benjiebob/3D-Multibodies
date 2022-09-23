"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join


base_dir = "/datasets/people_data"
IMG_RES = 224

H36M_ROOT = join(base_dir, "human36m")
LSP_ROOT = join(base_dir, "lsp")
LSP_ORIGINAL_ROOT = join(base_dir, "lsp_original")
LSPET_ROOT = join(base_dir, "hr-lspet")
MPII_ROOT = join(base_dir, "mpii")
COCO_ROOT = join(base_dir, "coco2014")
MPI_INF_3DHP_ROOT = join(base_dir, "mpi_inf_3dhp")
PW3D_ROOT = join(base_dir, "3dpw")
UPI_S1H_ROOT = join(base_dir, "upi-s1h-v2")

SPIN_ROOT = join(base_dir, "spin_fits")

# Output folder to save test/train npz files
BASE_FOLDER = "/scratch/code/3D-Multibodies"
DATASET_NPZ_PATH = join(BASE_FOLDER, "data/dataset_extras")
CROP_NPZ_PATH = join(BASE_FOLDER, "data/crops")

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate
# the .npz files with the annotations.
OPENPOSE_PATH = "datasets/openpose"

# Path to test/train npz files
DATASET_FILES = [
    {
        "h36m-p1": join(DATASET_NPZ_PATH, "h36m_valid_protocol1.npz"),
        "h36m-p2": join(DATASET_NPZ_PATH, "h36m_valid_protocol2.npz"),
        "h36m-p1-corr": join(DATASET_NPZ_PATH, "h36m_valid_protocol1.npz"),
        "lsp": join(DATASET_NPZ_PATH, "lsp_dataset_test.npz"),
        "mpi-inf-3dhp": join(DATASET_NPZ_PATH, "mpi_inf_3dhp_valid.npz"),
        "3dpw": join(
            BASE_FOLDER, "3dmb/3dpw_test_all.npz"
        ),  # This version contains openpose keypoints (required for A3DPW)
        "coco": join(
            DATASET_NPZ_PATH, "coco_val_amb_4_10.npz"
        ),  # Not distributed with SPIN data
    },
    {
        "h36m": join(
            DATASET_NPZ_PATH, "h36m_train.npz"
        ),  # Not distributed with SPIN data
        "lsp-orig": join(DATASET_NPZ_PATH, "lsp_dataset_original_train.npz"),
        "mpii": join(DATASET_NPZ_PATH, "mpii_train.npz"),
        "coco": join(DATASET_NPZ_PATH, "coco_2014_train.npz"),
        "lspet": join(DATASET_NPZ_PATH, "hr-lspet_train.npz"),
        "mpi-inf-3dhp": join(DATASET_NPZ_PATH, "mpi_inf_3dhp_train.npz"),
    },
]

# Not distributed with SPIN
CROP_FILES = {
    "h36m-p2": join(CROP_NPZ_PATH, "h36m-p2_amb3.npz"),
    "3dpw": join(CROP_NPZ_PATH, "3dpw_amb3.npz"),
}

DATASET_FOLDERS = {
    "h36m": H36M_ROOT,
    "h36m-p1": H36M_ROOT,
    "h36m-p1-corr": H36M_ROOT,
    "h36m-p2": H36M_ROOT,
    "lsp-orig": LSP_ORIGINAL_ROOT,
    "lsp": LSP_ROOT,
    "lspet": LSPET_ROOT,
    "mpi-inf-3dhp": MPI_INF_3DHP_ROOT,
    "mpii": MPII_ROOT,
    "coco": COCO_ROOT,
    "3dpw": PW3D_ROOT,
    "upi-s1h": UPI_S1H_ROOT,
}

DATASET_DICT = {
    "h36m": 0,
    "lsp-orig": 1,
    "mpii": 2,
    "lspet": 3,
    "coco": 4,
    "mpi-inf-3dhp": 5,
    "3dpw": 6,
}

DATASET_EVAL_KEYS = [0]

DELETE_TYPE = ["NONE", "LEGS", "ARMS+HEAD", "HEAD"]
DELETE_PROB = [0.1, 0.3, 0.3, 0.3]

DATASET_LIST = ["h36m", "lsp-orig", "mpii", "lspet", "coco", "mpi-inf-3dhp", "3dpw"]

CUBE_PARTS_FILE = join(BASE_FOLDER, "data/cube_parts.npy")
JOINT_REGRESSOR_TRAIN_EXTRA = join(BASE_FOLDER, "data/J_regressor_extra.npy")
JOINT_REGRESSOR_H36M = join(BASE_FOLDER, "data/J_regressor_h36m.npy")
JOINT_REGRESSOR_3DMB = join(
    BASE_FOLDER, "data/3dmb/J_regressor.npz"
)  # Not distributed with SPIN
VERTEX_TEXTURE_FILE = join(BASE_FOLDER, "data/vertex_texture.npy")
STATIC_FITS_DIR = join(BASE_FOLDER, "data/static_fits")
SMPL_MEAN_PARAMS = join(BASE_FOLDER, "data/smpl_mean_params.npz")
SMPL_MODEL_DIR = join(BASE_FOLDER, "data/smpl")

SPIN_FITS_DIR = join(BASE_FOLDER, "data/spin_fits")
HMR_PRETRAINED = join(BASE_FOLDER, "data/pretrained/model_checkpoint.pt")
