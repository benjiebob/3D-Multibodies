import torch
import numpy as np

import sys
import config
sys.path.append('../smplx')
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

from utils import constants

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


class SMPLEvaluator():
    def __init__():
        # Load SMPL model
        self.smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                            create_transl=False).to(device)
        self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                            gender='male',
                            create_transl=False).to(device)
        self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                            gender='female',
                            create_transl=False).to(device)

        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    # For predicted joints, always run smpl_neutral
    # For ground truth, run gendered smpl for datasets != h36m / mpi-inf
    # This method is always used for evaluation, so can be run with torch.no_grad() wrapper
    def run_batch(self, 
        dataset_to_run,
        pred_pose, pred_betas,
        gt_pose, gt_betas, gt_joints, 
        dataset_name, gender):

        pred_neutral = self.smpl_neutral(
            betas=pred_betas, 
            body_pose=pred_pose[:,1:], 
            global_orient=pred_pose[:,0].unsqueeze(1), 
            pose2rot=False)
        pred_vertices = pred_neutral.vertices

        gt_neutral = self.smpl_neutral(
            betas=gt_betas, 
            body_pose=gt_pose[:,1:], 
            global_orient=gt_pose[:,0].unsqueeze(1), 
            pose2rot=False)
        gt_vertices = pred_neutral.vertices

                        



        if gender is not None:
            dataset_idxs = dataset_name != 'h36m' or 'mpi-inf'
            if len(dataset_idxs) > 0:
                gt_vertices = self.smpl_male(
                    betas=betas, 
                    body_pose=rotmat[:,1:], 
                    global_orient=rotmat[:,0].unsqueeze(1), 
                    pose2rot=False).vertices 
                gt_vertices_female = self.smpl_female(
                    betas=betas, 
                    body_pose=rotmat[:,1:], 
                    global_orient=rotmat[:,0].unsqueeze(1), 
                    pose2rot=False).vertices
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                verts[dataset_idxs] = gt

        joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

        # Regressor broadcasting
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)

        # Get 14 predicted joints from the mesh
        keypoints_3d = torch.matmul(J_regressor_batch, verts)
        pelvis = keypoints_3d[:, [0],:].clone()
        keypoints_3d = keypoints_3d[:, joint_mapper_h36m, :]
        keypoints_3d = keypoints_3d - pelvis

        # Note: replace keypoints_3d for pose_3d if running ground truth with
        # 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
        # gt_keypoints_3d = batch['pose_3d'].cuda()
        # gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]

        return verts, keypoints_3d

