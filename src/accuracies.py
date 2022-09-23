import torch
import numpy as np
import config
from utils import constants
from utils.pose_utils import reconstruction_error
from networks.smpl.smpl_spin import SMPL

import trimesh

class AccuracyMetrics():
    def __init__(self):

        # Load SMPL model
        self.smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                            create_transl=False)

        self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                        gender='male',
                        create_transl=False)

        self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                        gender='female',
                        create_transl=False)

        self.J_regressor = torch.from_numpy(
            np.load(config.JOINT_REGRESSOR_H36M)).float()

    # DATASET_DICT = {
    #   'h36m': 0, 
    #   'lsp-orig': 1, 
    #   'mpii': 2, 
    #   'lspet': 3, 
    #   'coco': 4, 
    #   'mpi-inf-3dhp': 5
    # }
    def compute_accuracies(self, 
        pred_pose, pred_betas,
        gt_pose, gt_betas,
        gt_joints, gender, 
        dataset_keys,
        dataset_eval_list):

        result_dict = {}
        for dataset_eval_key in dataset_eval_list:
            dataset_mask = (dataset_keys == dataset_eval_key)
            if dataset_mask.shape[0] == 0 or torch.sum(dataset_mask) == 0:
                result_dict[dataset_eval_key] = None # no results for this dataset
            else:
                result_dict[dataset_eval_key] = self.run_dataset(
                    dataset_eval_key,
                    pred_pose[dataset_mask], pred_betas[dataset_mask],
                    gt_pose[dataset_mask], gt_betas[dataset_mask],
                    gt_joints[dataset_mask], gender[dataset_mask])

        return result_dict

    def get_mappers(self, dataset_key):
        if dataset_key == 5:
            joint_mapper_h36m = constants.H36M_TO_J17
            joint_mapper_gt = constants.J24_TO_J17
        else:
            joint_mapper_h36m = constants.H36M_TO_J14
            joint_mapper_gt = constants.J24_TO_J14

        if dataset_key == 0 or dataset_key == 5:
            use_gendered_smpl = False
        else:
            use_gendered_smpl = True

        return joint_mapper_h36m, joint_mapper_gt, use_gendered_smpl


    def compute_all_pairs(self, pred_joints, eval_mode_ids):
        batch_size = pred_joints.shape[0]
        num_modes = eval_mode_ids.shape[1]

        eval_quantized_joints = []
        for i in range(batch_size):
            eval_quantized_joints.append(
                pred_joints[i, eval_mode_ids[i]])
        eval_quantized_joints = torch.stack(eval_quantized_joints, dim = 0)

        # B, M, J, 3 -> B x J, M, 3
        eval_quantized_rs = eval_quantized_joints.permute(0, 2, 1, 3).reshape(-1, num_modes, 3).contiguous()
        
        # -> B x J, M, M
        joints_allpairs = torch.cdist(eval_quantized_rs, eval_quantized_rs)
        joints_allpairs = joints_allpairs.reshape(batch_size, -1, num_modes, num_modes)
        
        # -> B, M, M
        joints_allpairs = torch.mean(joints_allpairs, dim = 1)

        lower_els = []
        for j in range(batch_size):
            lower_els.append(
                torch.mean(torch.tril(joints_allpairs[j], diagonal=-1)) * 1000.0)
        
        return torch.stack(lower_els, dim = 0)

    def run_dataset(self, dataset_key,
        pred_pose, pred_betas,
        gt_pose, gt_betas, 
        gt_joints, gender):

        joint_mapper_h36m, joint_mapper_gt, use_gendered_smpl = self.get_mappers(
            dataset_key
        )

        # Regressor broadcasting
        J_regressor_batch = self.J_regressor[None, :].to(pred_pose.device).expand(
            pred_pose.shape[0], -1, -1)

        # print ("BETAS: {0}, POSE: {1}".format(pred_betas.device, pred_pose.device))

        pred_neutral = self.smpl_neutral.to(pred_pose.device)(
            betas=pred_betas, 
            body_pose=pred_pose[:,1:], 
            global_orient=pred_pose[:,0].unsqueeze(1), 
            pose2rot=False)
        pred_vertices = pred_neutral.vertices

        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        gt_neutral = self.smpl_neutral.to(pred_pose.device)(
            betas=gt_betas, 
            body_pose=gt_pose[:,1:], 
            global_orient=gt_pose[:,0].unsqueeze(1), 
            pose2rot=False)
        gt_vertices = gt_neutral.vertices

        # Get 14 ground truth joints
        # TODO: Implemented thread safe eval
        # assert not use_gendered_smpl, "Haven't implemented gendered SMPL validation"
        if use_gendered_smpl == False:
            gt_keypoints_3d = gt_joints
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]     
            
        # For 3DPW get the 14 common joints from the rendered shape
        else:
            gt_vertices = self.smpl_male.to(pred_pose.device)(
                betas=gt_betas, 
                body_pose=gt_pose[:,1:], 
                global_orient=gt_pose[:,0].unsqueeze(1), 
                pose2rot=False).vertices 
            gt_vertices_female = self.smpl_female.to(pred_pose.device)(
                betas=gt_betas, 
                body_pose=gt_pose[:,1:], 
                global_orient=gt_pose[:,0].unsqueeze(1), 
                pose2rot=False).vertices

            gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1)
        
        # Reconstuction_error
        try:
            r_error = reconstruction_error(
                pred_keypoints_3d.data.cpu().numpy(), 
                gt_keypoints_3d.data.cpu().numpy(), reduction=None)

            r_error = torch.from_numpy(r_error) * 1000.0
        except:
            r_error = None

        # vertex error
        gt_pelvis2 = torch.matmul(J_regressor_batch, gt_vertices)[:, [0],:].clone()

        pred_verts_norm = pred_vertices - pred_pelvis
        gt_vertices_norm = gt_vertices - gt_pelvis2
        s_error_norm = torch.sqrt(((pred_verts_norm - gt_vertices_norm) ** 2).sum(dim=-1)).mean(dim=-1) * 1000.0

        try:
            s_error_reco = reconstruction_error(
                pred_vertices.data.cpu().numpy(), 
                gt_vertices.data.cpu().numpy(), reduction=None)

            s_error_reco = torch.from_numpy(s_error_reco) * 1000.0
        except:
            s_error_reco = None

        metrics = {
            'mpjpe' : error * 1000.0,
            'r_error' : r_error,
            's_error' : s_error_reco,
        }

        result_data = {
            'gt_vertices' : gt_vertices_norm,
            'pred_vertices' : pred_verts_norm,
            'gt_keypoints_EVAL' : gt_keypoints_3d,
            'pred_keypoints_EVAL' : pred_keypoints_3d,
        }

        return metrics, result_data


    
