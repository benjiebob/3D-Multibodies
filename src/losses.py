
import torch
import torch.nn as nn
from utils.geometry import batch_rodrigues
import numpy as np

class Losses():
    def __init__(self):
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss(reduction='none')
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss(reduction='none')
        
    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        assert pred_keypoints_2d.shape[0] == \
            gt_keypoints_2d.shape[0], "Batch sizes don't match"

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(
            pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).reshape(
                pred_keypoints_2d.shape[0], -1).mean(dim=-1)
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        assert pred_keypoints_3d.shape[0] == \
            gt_keypoints_3d.shape[0] == \
                has_pose_3d.shape[0], "Batch sizes don't match"

        keypoint_error_rtn = torch.zeros(has_pose_3d.shape[0]).to(has_pose_3d.device)

        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            keypoint_error = (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d))
            
            keypoint_error_rtn[has_pose_3d==1] = keypoint_error.reshape(
                keypoint_error.shape[0], -1).mean(dim=-1)
            
        return keypoint_error_rtn

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        assert pred_vertices.shape[0] == \
            gt_vertices.shape[0] == \
                has_smpl.shape[0], "Batch sizes don't match"

        shape_loss_rtn = torch.zeros(has_smpl.shape[0]).to(has_smpl.device)

        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            shape_error = self.criterion_shape(
                pred_vertices_with_shape, 
                gt_vertices_with_shape).reshape(has_smpl.shape[0], -1).mean(dim=-1)
            shape_loss_rtn[has_smpl==1] = shape_error
        return shape_loss_rtn
        
    def smpl_losses(self, pred_rotmat, pred_betas, gt_rotmat, gt_betas, has_smpl):
        assert pred_rotmat.shape[0] == \
            pred_betas.shape[0] == \
                gt_rotmat.shape[0] == \
                    gt_betas.shape[0] == \
                        has_smpl.shape[0], "Batch sizes don't match"
        
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        
        # bjb_edit: Do L2 difference between axis angle representations.
        gt_rotmat_valid = gt_rotmat[has_smpl == 1]
        # gt_rotmat_valid = gt_pose[has_smpl == 1]

        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            # loss_regr_pose = self.criterion_regr(
            #     pred_rotmat_valid[:, 1:], gt_rotmat_valid[:, 1:]).reshape(has_smpl.shape[0], -1).mean(dim=-1)
            loss_regr_pose = self.criterion_regr(
                pred_rotmat_valid, gt_rotmat_valid).reshape(has_smpl.shape[0], -1).mean(dim=-1)
            loss_regr_betas = self.criterion_regr(
                pred_betas_valid, gt_betas_valid).reshape(has_smpl.shape[0], -1).mean(dim=-1)
        else:
            loss_regr_pose = torch.zeros(has_smpl.shape[0]).to(has_smpl.device)
            loss_regr_betas = torch.zeros(has_smpl.shape[0]).to(has_smpl.device)
        return loss_regr_pose, loss_regr_betas

    '''
    This function picks the min-of-n mode based on 3D training data.
    It picks using the GT sensor data (gt_keypoints_3d), but in cases this doesn't exist
    reverts back to the GT SMPL data.
    If neither of this exist, there is a return value (probably random) but
    this should be when has_pose_3d or has_smpl are applied in the losses later on.
    '''
    def select_modes(self, 
        pred_keypoints_3d, 
        gt_keypoints_3d, 
        gt_model_joints,
        has_pose_3d,
        has_smpl,
        num_modes,
        return_median=False):

        assert pred_keypoints_3d.shape[0] == \
            gt_keypoints_3d.shape[0] == \
                gt_model_joints.shape[0] == \
                    has_pose_3d.shape[0] == \
                        has_smpl.shape[0], "Batch size doesn't match"
                

        # Compute 3D loss against sensor data
        # Invisble & has_pose_3d values are set to 0
        keypoints_3d_acc = self.keypoint_3d_loss(
            pred_keypoints_3d, gt_keypoints_3d, has_pose_3d)
            
        # All 3D points for SMPL are visible (it's a mesh!) so 
        # visibility not needed
        keypoints_3d_acc_model = self.criterion_keypoints(
            pred_keypoints_3d[:, 25:, :], 
            gt_model_joints[:, 25:, :]).reshape(
                pred_keypoints_3d.shape[0], -1).mean(dim=-1)
        
        accuracy_bymode = torch.zeros(
            pred_keypoints_3d.shape[0]).to(keypoints_3d_acc.device)
        
        accuracy_bymode[has_smpl==1] = keypoints_3d_acc_model[has_smpl==1]
        accuracy_bymode[has_pose_3d==1] = keypoints_3d_acc[has_pose_3d==1]
        
        # Compute the best mode, according to 3D joint loss
        accuracy_bymode = accuracy_bymode.reshape(-1, num_modes)
        if return_median:
            minofn_ids = torch.argsort(accuracy_bymode.detach(), dim = 1)[:, num_modes // 2]
        else:
            minofn_ids = torch.argmin(accuracy_bymode.detach(), dim = 1)
        return minofn_ids, accuracy_bymode

    def mean_log_Gaussian_like(self, means, sigma, alpha, y_true, num_modes, return_exponent=False):
        """Mean Log Gaussian Likelihood distribution
        y_truth: ground truth 3d pose
        parameters: output of hypotheses generator, which conclude the mean, variance and mixture coeffcient of the mixture model
        c: dimension of 3d pose
        m: number of kernels
        """

        batch_size = means.shape[0]

        means = means.view(batch_size, num_modes, -1)
        sigma = torch.clamp(sigma, 1e-15, 1e15)
        alpha = torch.clamp(alpha, 1e-8, 1.)

        y_true = y_true.view(batch_size, -1)

        c = means.shape[-1]

        exponent = torch.log(alpha) - 0.5 * c * torch.log(torch.FloatTensor([2 * np.pi]).cuda()) \
                        - c * torch.log(sigma) \
                        - torch.sum((y_true.unsqueeze(1) - means) ** 2, dim = -1) / (2.0 * (sigma) ** 2.0)

        if return_exponent:
            return exponent
        else:
            log_gauss = torch.logsumexp(exponent, dim=1)
            return log_gauss