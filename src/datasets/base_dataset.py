from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join
import os
import io

import config
from utils import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import tarfile
from PIL import Image


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(
        self,
        options,
        dataset,
        dataset_key=None,
        ambiguous=False,
        ignore_3d=False,
        use_augmentation=True,
        is_train=True,
        allow_static_fits=False,
        run_mini=False,
        return_img_orig=False,
        ignore_img=False,
    ):

        super(BaseDataset, self).__init__()
        print("->-> LOADING DATA FROM: {0}".format(config.base_dir))

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.ambiguous = ambiguous
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(
            mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD
        )
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data["imgname"]
        self.run_mini = run_mini
        self.allow_static_fits = allow_static_fits
        self.return_img_orig = return_img_orig
        self.ignore_img = ignore_img

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data["maskname"]
        except KeyError:
            pass
        try:
            self.partname = self.data["partname"]
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data["scale"]
        self.center = self.data["center"]

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data["pose"].astype(np.float)
            self.betas = self.data["shape"].astype(np.float)
            if "has_smpl" in self.data:
                self.has_smpl = self.data["has_smpl"]
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
            self.pose = None
            self.betas = None
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data["S"]
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get 2D keypoints
        try:
            if self.dataset == "3dpw":  # hack used for my 3dpw npz files
                keypoints_gt = self.data["parts_openpose_s24"]
            else:
                keypoints_gt = self.data["part"]
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data["openpose"]
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # TODO(Ben): Ensure this is read from disk
        if self.ambiguous:
            if self.is_train:
                try:
                    self.delete_type = self.data["delete_type"]
                    print("Loading delete type from disk for: {0}".format(self.dataset))
                except KeyError:           
                    self.delete_type = np.random.choice(
                        len(config.DELETE_PROB), size=len(self.imgname), p=config.DELETE_PROB
                    )
            else:
                if dataset in config.CROP_FILES:
                    amb_data = np.load(config.CROP_FILES[dataset])
                else:
                    raise Exception(f"No precomputed crops available for dataset: {dataset}")

                self.amb_center = amb_data["amb_center"]
                self.amb_scale = amb_data["amb_scale"]
                self.delete_type = np.ones(len(self.imgname)) * -1.0 # We don't have delete_types for precomputed crops
        else:
            self.delete_type = np.ones(len(self.imgname)) * -1.0 # Non-ambiguous should not use delete_type
        

        # Get gender data, if available
        try:
            gender = self.data["gender"]
            self.gender = np.array([0 if str(g) == "m" else 1 for g in gender]).astype(
                np.int32
            )
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        self.has_static = np.zeros_like(self.has_smpl)

        fits_path = join(config.STATIC_FITS_DIR, "{0}_fits.npy".format(dataset))
        if os.path.exists(fits_path) and self.allow_static_fits:
            static_fits = np.load(fits_path)
            if self.pose is None:
                self.pose = np.zeros((self.has_smpl.shape[0], 72))
                self.betas = np.zeros((self.has_smpl.shape[0], 10))

            for index in range(self.has_smpl.shape[0]):
                if not self.has_smpl[index]:
                    self.pose[index] = static_fits[index, :72]
                    self.betas[index] = static_fits[index, 72:]
                    self.has_smpl[index] = 1.0
                    self.has_static[index] = 1.0
        else:
            self.static_fits = None

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(
                1 - self.options.noise_factor, 1 + self.options.noise_factor, 3
            )

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(
                2 * self.options.rot_factor,
                max(
                    -2 * self.options.rot_factor,
                    np.random.randn() * self.options.rot_factor,
                ),
            )

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + self.options.scale_factor,
                max(
                    1 - self.options.scale_factor,
                    np.random.randn() * self.options.scale_factor + 1,
                ),
            )
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(
            rgb_img, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot
        )
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(
                kp[i, 0:2] + 1,
                center,
                scale,
                [constants.IMG_RES, constants.IMG_RES],
                rot=r,
            )
        # convert to normalized coordinates
        kp[:, :-1] = 2.0 * kp[:, :-1] / constants.IMG_RES - 1.0
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype("float32")
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum("ij,kj->ki", rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype("float32")
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype("float32")
        return pose

    def compute_crop_from_deletetype(self, delete_type, keypoints):
        DELETE_TYPE = ['NONE', 'LEGS', 'ARMS+HEAD', 'HEAD']
        
        vis_indices = keypoints[:, -1]
        joints_to_delete = []

        if delete_type > 0:
            if delete_type == 1:  # Legs
                joints_to_delete += [0, 1, 2]  # delete right leg
                joints_to_delete += [3, 4, 5]  # delete left leg
            elif delete_type == 2:  # Arms+Head
                joints_to_delete += [6, 7, 8]  # delete right arms
                joints_to_delete += [9, 10, 11]  # delete left arms
                joints_to_delete += [
                    12,
                    13,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                ]  # delete head + neck
            elif delete_type == 3:
                joints_to_delete += [6, 7, 8]  # delete right arms
                joints_to_delete += [9, 10, 11]  # delete left arms

            joints_to_delete = [x + 25 for x in joints_to_delete]
            joints_to_delete += range(25)

            vis_indices[joints_to_delete] = 0
            vis_indices = vis_indices.astype(bool)

            kps_to_wrap = keypoints[vis_indices, :2]
            bbox = np.array(
                [
                    np.min(kps_to_wrap[:, 0]),
                    np.min(kps_to_wrap[:, 1]),
                    np.max(kps_to_wrap[:, 0]) + 1,
                    np.max(kps_to_wrap[:, 1]) + 1,
                ]
            )

            center = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2])
            scale = np.array(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0)

            invis = (1 - np.array(vis_indices)).astype(bool)
            keypoints[invis, :-1] = 0.0  # Zero the joints
            keypoints[invis, 2] = 0.0  # Zero the visibility flag

            return center, scale, keypoints

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        item["scale_orig"] = float(sc * scale)
        item["center_orig"] = center.astype(np.float32)

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            pose_3d = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            assert pose_3d[2, -1] == pose_3d[3, -1] == 1, "Pelvis joints are missing"
            pelvis = (pose_3d[2, :3] + pose_3d[3, :3])[None, :] / 2
            pose_3d[:, :3] = pose_3d[:, :3] - pelvis
            item["pose_3d"] = pose_3d
        else:
            item["pose_3d"] = torch.zeros(24, 4, dtype=torch.float32)

        keypoints = self.keypoints[index].copy()

        if self.ambiguous:
            if self.is_train:
                delete_type = self.delete_type[index].copy()
                if keypoints[:, -1].sum() > 0.0:
                    center, scale, keypoints = self.compute_crop_from_deletetype(delete_type, keypoints)
            else:
                # If eval, must load delete types from disk
                center = self.amb_center[index]
                scale = self.amb_scale[index]

        # Get 2D keypoints and apply augmentation transforms
        item["keypoints"] = torch.from_numpy(
            self.j2d_processing(keypoints, center, sc * scale, rot, flip)
        ).float()

        img_basename = self.imgname[index]
        try:
            img_basename = img_basename.decode()
        except (UnicodeDecodeError, AttributeError):
            pass

        imgname = join(self.img_dir, img_basename)

        if not self.ignore_img:
            # Process image
            try:
                img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)
            except TypeError:
                print("Error loading image: {0}".format(imgname))

            img_orig = img.copy()
            img = self.rgb_processing(img, center, sc * scale, rot, flip, pn)
            img = torch.from_numpy(img).float()

            # Store image before normalization to use it in visualization
            item["img"] = self.normalize_img(img)

            if self.return_img_orig:
                item["img_orig"] = torch.from_numpy(
                    np.transpose(img_orig, (2, 0, 1))
                ).float()

        item["pose"] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item["betas"] = torch.from_numpy(betas).float()
        item["imgname"] = imgname

        item["has_smpl"] = self.has_smpl[index]
        item["has_static"] = self.has_static[index]
        item["has_pose_3d"] = self.has_pose_3d
        item["scale"] = float(sc * scale)
        item["center"] = center.astype(np.float32)
        item["is_flipped"] = flip
        item["rot_angle"] = np.float32(rot)
        item["gender"] = self.gender[index]
        item["sample_index"] = index
        if self.dataset_key is not None:
            item["dataset_key"] = self.dataset_key
        else:
            item["dataset_key"] = config.DATASET_DICT[self.dataset]

        return item

    def __len__(self):
        if self.run_mini:
            return 100
        else:
            return len(self.imgname)
