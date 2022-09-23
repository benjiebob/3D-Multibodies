import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import time

from exp_manager.base_model import BaseModel
from exp_manager.utils import auto_init_args
from exp_manager.config import get_default_args

import os
import networks.smpl.smpl as smpl
from networks.spin import ConditionalHMR
from networks.benvp.benvp import RealNVP
import torchgeometry as tgm

from losses import Losses
from accuracies import AccuracyMetrics
import config as global_config
from utils import constants
from utils.kmeans import GPUKmeans
from utils.skeleton import draw_skeleton

from utils.geometry import perspective_projection, batch_rodrigues
from utils.minofn import MinofN
from visdom_plotly import VisdomPlotly

class Model(BaseModel):
    def __init__( self, 
            loss_weights={
                'loss_joint':25.0, 'loss_angle':1.0, 'loss_shape':0.001,
                'loss_skel2d':1.0, 'loss_vertex':0.0, 
                'loss_nll':1e-3, 'loss_nll_reg':1e-3,
                'loss_skel2d_modewise':0.1, 'loss_depth_modewise':1.0
            },
            num_modes = 25,
            init_flow="",
            COND_HMR = get_default_args(ConditionalHMR),
            openpose_train_weight=0.0,
            gt_train_weight=1.0,
            log_vars=[
                'objective',
                'loss_nll', 'loss_nll_reg',
                'loss_joint','loss_angle', 'loss_shape', 
                'loss_skel2d', 'loss_skel2d_modewise', 'loss_vertex',
                'acc_h36m_M01_mpjpe',
                'acc_h36m_M05_mpjpe',
                'acc_h36m_WM05_mpjpe',
                'acc_h36m_M10_mpjpe',
                'acc_h36m_WM10_mpjpe',
                'acc_h36m_M25_mpjpe',
                'acc_h36m_WM25_mpjpe',
                'acc_h36m_M100_mpjpe',
                'dur_acc', 'dur_flow', 'dur_hmr', 'dur_losses',
                'dur_minofn', 'dur_smpl'
            ], 
            **kwargs):

        auto_init_args(self)

        super(Model, self).__init__(
            loss_weights=loss_weights, 
            log_vars=log_vars)

        self.focal_length = constants.FOCAL_LENGTH
        d = 24 * 3
        n_hidden = 512
        num_transformations = 20
        dropout = 0.2

        self.num_dims_flow = d

        prior_mean = torch.zeros(d)
        prior_var = torch.eye(d)
        masks = (torch.rand(num_transformations,d)<0.5).float()
        nets = lambda: nn.Sequential(
            nn.Linear(d, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(), 
            nn.Linear(n_hidden, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(), 
            nn.Linear(n_hidden, d), nn.Tanh())
        
        nett = lambda: nn.Sequential(
            nn.Linear(d, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(), 
            nn.Linear(n_hidden, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(), 
            nn.Linear(n_hidden, d))

        self.model_realnvp = RealNVP(nets, nett, masks, prior_mean, prior_var)
        self.model = ConditionalHMR(
            num_modes, 
            num_transformations,
            **COND_HMR)

        self.accuracy_metrics = AccuracyMetrics()

        print ("----- SPIN Layers to Optimize -----")
        for name_s, param_s in self.model.named_parameters():
            if param_s.requires_grad:
                print (name_s)

        neutral_model_path = os.path.join(global_config.SMPL_MODEL_DIR, 'SMPL_NEUTRAL.pkl')
        self.smpl_mini = smpl.SMPL(neutral_model_path)
        self.losses = Losses()

    
    def forward(self, epoch, input_batch, cache_mode=False):   
        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose

        gender = input_batch['gender']

        has_smpl = input_batch['has_smpl'].bool() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].bool() # flag that indicates whether 3D pose is valid

        dataset_key = input_batch['dataset_key']
        batch_size, _, _, _ = images.shape
        device = images.device

        gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(
            batch_size, 24, 3, 3)

        gt_vertices, gt_model_joints, gt_pelvis = self.smpl_mini(
            gt_rotmat, gt_betas, run_mini = True)

        log_liklihood = torch.zeros(batch_size).to(device)
        log_liklihood_reg = torch.zeros(batch_size, self.num_modes).to(device)
    
        hmr_pred = self.model(images)
        pred_shape = hmr_pred['pred_shape']
        pred_camera = hmr_pred['pred_camera']
        pred_global_rotmat = hmr_pred['pred_global_rotmat']
        pred_pose_axis = MinofN.compress_modes_into_batch(
            hmr_pred['pred_local_axis'])

        if torch.sum(has_smpl) > 0:
            # TODO: undo the rotation
            gt_rotmat_zero = gt_rotmat.clone()
            gt_rotmat_zero[:, 0] = torch.eye(3).to(gt_rotmat.device)
            _, gt_model_joints_zero, _ = self.smpl_mini(
                gt_rotmat_zero, gt_betas, run_mini = True)
            joints_gt_norm = gt_model_joints_zero[has_smpl==1, 25:, ].clone()
            logl, _ = self.model_realnvp.log_prob(
                joints_gt_norm.reshape(-1, self.num_dims_flow))
            log_liklihood[has_smpl==1] = logl

        pred_rotmat = batch_rodrigues(pred_pose_axis.contiguous().view(-1,3)).view(
            batch_size, self.num_modes, 23, 3, 3)

        # out_fpose_mode: (N, M, 24, 3, 3)
        out_fpose_mode = torch.cat([pred_global_rotmat, pred_rotmat], dim = 2)

        # Run all modes through SMPL, after compressing 
        # (N, M, ...) -> (N * M, ...)
        out_fpose_mode_compressed = MinofN.compress_modes_into_batch(out_fpose_mode)
        out_shape_compressed = MinofN.compress_modes_into_batch(pred_shape)
        out_verts_mode, out_model_joints_mode, out_model_pelvis = self.smpl_mini(
            out_fpose_mode_compressed,
            out_shape_compressed,
            run_mini = True)

        out_fpose_mode_compressed_zero = out_fpose_mode_compressed.clone()
        out_fpose_mode_compressed_zero[:, 0] = torch.eye(3).to(
            out_fpose_mode_compressed_zero.device)
        _, out_model_joints_mode_zero, _ = self.smpl_mini(
            out_fpose_mode_compressed_zero,
            out_shape_compressed,
        run_mini = True)

        joints_norm = out_model_joints_mode_zero.detach()[:, 25:, :].clone()
        
        for p in self.model_realnvp.parameters():
            p.requires_grad = False
        pred_reg, _ = self.model_realnvp.log_prob(
            joints_norm.reshape(-1, self.num_dims_flow)
        )
        for p in self.model_realnvp.parameters():
            p.requires_grad = True
            
        log_liklihood_reg = MinofN.decompress_modes_from_batch(
            pred_reg, batch_size, self.num_modes)

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        out_camera_mode = MinofN.compress_modes_into_batch(pred_camera)
        tz = 2*self.focal_length/(global_config.IMG_RES * out_camera_mode[:,0] +1e-9)
        pred_cam_t = torch.stack([out_camera_mode[:,1],
                                  out_camera_mode[:,2],
                                  tz],dim=-1)

        camera_center = torch.zeros(batch_size * self.num_modes, 2)
        camera_rotation = torch.eye(3).unsqueeze(0).expand(
            batch_size * self.num_modes, -1, -1).to(device)

        loss_depth_all = ((torch.exp(-out_camera_mode[:,0]*10)) ** 2).reshape(
            batch_size, -1).mean(dim=-1)

        # (N * M, 49, 2)
        projected_joints_mode = perspective_projection(
            out_model_joints_mode + out_model_pelvis,
            rotation=camera_rotation,
            translation=pred_cam_t,
            focal_length=self.focal_length,
            camera_center=camera_center)

        # Normalize keypoints to [-1,1]
        projected_joints_mode = projected_joints_mode / (global_config.IMG_RES / 2.)
      
        # Expand the gt_keypoints
        loss_skel2d_all = self.losses.keypoint_loss(
            projected_joints_mode, 
            MinofN.expand_and_compress_modes(gt_keypoints_2d, self.num_modes),
            self.openpose_train_weight, self.gt_train_weight
        ).reshape(batch_size, -1).mean(dim = -1)

        ######## Identfy and select the min-of-n modes ########
        mean_ids = torch.zeros(batch_size, dtype=int).to(device)
        if self.num_modes > 1:
            minofn_mode_ids, _ = self.losses.select_modes(
                out_model_joints_mode, # Predicted joints (N * M, 49, 3)
                MinofN.expand_and_compress_modes(gt_joints, self.num_modes), # GT Sensor data
                MinofN.expand_and_compress_modes(gt_model_joints, self.num_modes), # GT SMPL data
                MinofN.expand_and_compress_modes(has_pose_3d, self.num_modes),
                MinofN.expand_and_compress_modes(has_smpl, self.num_modes),
                self.num_modes
            )
        else:
            minofn_mode_ids = mean_ids
            
        out_model_joints_reshape = MinofN.decompress_modes_from_batch(
            out_model_joints_mode, batch_size, self.num_modes)

        out_verts_mode_reshape = MinofN.decompress_modes_from_batch(
            out_verts_mode, batch_size, self.num_modes)

        out_pelvis_mode_reshape = MinofN.decompress_modes_from_batch(
            out_model_pelvis, batch_size, self.num_modes)

        out_projected_joints_reshape = MinofN.decompress_modes_from_batch(
            projected_joints_mode, batch_size, self.num_modes)

        assert out_fpose_mode.shape[0] == \
            pred_camera.shape[0] == \
                pred_shape.shape[0] == \
                    out_verts_mode_reshape.shape[0] == \
                        out_model_joints_reshape.shape[0] == \
                            out_projected_joints_reshape.shape[0], "Batch sizes don't match"
        
        preds = {}
        preds['loss_joint'] = torch.zeros(batch_size).to(device)
        preds['loss_angle'] = torch.zeros(batch_size).to(device)
        preds['loss_shape'] = torch.zeros(batch_size).to(device)
        preds['loss_vertex'] = torch.zeros(batch_size).to(device)
        preds['loss_skel2d'] = torch.zeros(batch_size).to(device)

        for minofn_ids in [minofn_mode_ids, mean_ids]:
            batch_ids = np.arange(batch_size)
            mofn_fpose = out_fpose_mode[batch_ids, minofn_ids]
            mofn_betas = pred_shape[batch_ids, minofn_ids]

            mofn_verts = out_verts_mode_reshape[batch_ids, minofn_ids]
            mofn_model_joints = out_model_joints_reshape[batch_ids, minofn_ids]
            mofn_proj_joints = out_projected_joints_reshape[batch_ids, minofn_ids]
            ######## Compute losses ########

            # 2D reprojection loss for the 'selected' mode
            # Only apply this loss when that mode can be picked, i.e. when there
            # is some 3D training data to use.
            valid_3d_gt = has_smpl | has_pose_3d

            loss_skel2d_mofn = self.losses.keypoint_loss(
                mofn_proj_joints[valid_3d_gt], 
                gt_keypoints_2d[valid_3d_gt], 
                self.openpose_train_weight, 
                self.gt_train_weight)

            preds['loss_skel2d'] += loss_skel2d_mofn

            # SMPL parameter losses for 'selected' mode
            loss_angle, loss_shape = self.losses.smpl_losses(
                mofn_fpose, mofn_betas, 
                gt_rotmat, gt_betas, 
                has_smpl)
            
            preds['loss_angle'] += loss_angle
            preds['loss_shape'] += loss_shape

            # 3D keypoint loss
            preds['loss_joint'] += self.losses.keypoint_3d_loss(
                mofn_model_joints, gt_joints, 
                has_pose_3d)

            # Loss on the overall shape.
            # Consider running SMPL again with the selected pose & shape to 
            # obtain a full vertex loss. 
            # Subsampling is probably fine though
            
            preds['loss_vertex'] += self.losses.shape_loss(
                mofn_verts, gt_vertices, has_smpl
            )

        preds['gt_rotmat'] = gt_rotmat
        preds['gt_pelvis'] = gt_pelvis
        preds['log_weights'] = log_liklihood_reg

        preds['loss_skel2d_modewise'] = loss_skel2d_all
        preds['loss_nll'] = -1 * log_liklihood
        preds['loss_nll_reg'] = torch.zeros_like(log_liklihood)
        preds['loss_depth_modewise'] = loss_depth_all

        ######## Compute accuracy metrics ########

        with torch.no_grad():
            if cache_mode:
                dataset_eval_keys = [ float(dataset_key[0]) ]
            else:
                dataset_eval_keys = global_config.DATASET_EVAL_KEYS
            
            if not self.training:
                use_weight = {
                    1: [False]
                }

                for extras in [5, 10, 25]:
                    if extras <= self.num_modes:
                        use_weight[extras] = [True, False]
            else:
                use_weight = {
                    self.num_modes: [False]
                }

            for eval_mode, weight_list in use_weight.items():
                for use_weight in weight_list:
                    if eval_mode == 1:
                        eval_mode_ids = mean_ids
                        eval_fpose = out_fpose_mode[:, 0]
                        eval_betas = pred_shape[:, 0]
                    elif eval_mode == self.num_modes:
                        eval_mode_ids = minofn_mode_ids
                        eval_fpose = out_fpose_mode[batch_ids, minofn_mode_ids]
                        eval_betas = pred_shape[batch_ids, minofn_mode_ids]
                    else:
                        out_model_joints_vector = out_model_joints_reshape[:, :, 25:]
                        out_model_pelvis = (out_model_joints_vector[:, :, 2, :] + out_model_joints_vector[:, :, 3,:]) / 2
                        out_model_joints_vector = out_model_joints_vector - out_model_pelvis[:, :, None, :]

                        log_weights = log_liklihood_reg
                        if not use_weight:
                            log_weights = torch.zeros_like(log_weights)
                        
                        # Cluster M-1 modes, and add in the mean (M=0)
                        start_kmeans = time.time()
                        km = GPUKmeans(K=eval_mode-1)
                        res = km(
                            out_model_joints_vector.reshape(
                                batch_size, self.num_modes, -1)[:, 1:], 
                            log_W = log_weights[:, 1:], 
                            verbose=False)

                        end_kmeans = time.time()
                        kmeans_time = end_kmeans - start_kmeans
                        print ("KMEANS TIME: {0}, {1}".format(kmeans_time, eval_mode))

                        representatives_idx = res['representatives_idx'] + 1
                        representatives_idx = torch.cat([
                            mean_ids[:, None], representatives_idx], dim = 1)

                        out_model_joints_rep = torch.gather(
                            out_model_joints_reshape, 
                            1, 
                            representatives_idx[:, :, None, None].expand(
                                -1, -1, *out_model_joints_reshape.shape[2:]))

                        eval_mode_ids = representatives_idx
                        minofn_ids_eval, accuracy_ids = self.losses.select_modes(
                            MinofN.compress_modes_into_batch(out_model_joints_rep), # Predicted joints (N * M, 49, 3)
                            MinofN.expand_and_compress_modes(gt_joints, eval_mode), # GT Sensor data
                            MinofN.expand_and_compress_modes(gt_model_joints, eval_mode), # GT SMPL data
                            MinofN.expand_and_compress_modes(has_pose_3d, eval_mode),
                            MinofN.expand_and_compress_modes(has_smpl, eval_mode),
                            eval_mode
                        )

                        eval_mode_id = torch.gather(
                            representatives_idx,
                            1,
                            minofn_ids_eval[:, None]
                        )
                        eval_fpose = torch.gather(
                            out_fpose_mode,
                            1,
                            eval_mode_id[:,:,None,None,None].expand(
                                -1, -1, *out_fpose_mode.shape[2:]
                            )
                        )[:, 0]
                        eval_betas = torch.gather(
                            pred_shape,
                            1,
                            eval_mode_id[:,:,None].expand(
                                -1, -1, *pred_shape.shape[2:]
                            )
                        )[:, 0]

                    results_all_datasets = self.accuracy_metrics.compute_accuracies(
                        eval_fpose, eval_betas,
                        gt_rotmat, gt_betas,
                        gt_joints, gender,
                        dataset_key, dataset_eval_keys)

                    if use_weight:
                        use_weight_str = "W"
                    else:
                        use_weight_str = ""

                    eval_name = "{0}M{1:02d}_ids".format(
                        use_weight_str, eval_mode)
                    preds[eval_name] = eval_mode_ids

                    for d_name, results in results_all_datasets.items():
                        if results is not None:
                            metrics, result_data = results
                            for metric_name, metric_val in metrics.items():
                                if metric_val is not None:
                                    if cache_mode:
                                        acc_name = "{0}M{1:02d}_{2}".format(
                                            use_weight_str, eval_mode, metric_name)
                                    else:
                                        acc_name = "acc_{0}_{1}M{2:02d}_{3}".format(
                                            global_config.DATASET_LIST[d_name], 
                                            use_weight_str,
                                            eval_mode,
                                            metric_name)

                                    preds[acc_name] = metric_val

                        if cache_mode:
                            for result_name, result_datum in result_data.items():
                                assert result_datum.shape[0] == batch_size, "Cache dataset should be of a single dataset type"
                                preds[result_name] = result_datum


        preds['out_joints_mode'] = out_model_joints_reshape
        preds['out_verts_mode'] = out_verts_mode_reshape
        preds['out_pelvis_mode'] = out_pelvis_mode_reshape
        preds['out_fpose_mode'] = out_fpose_mode
        preds['out_shape_mode'] = pred_shape
        preds['out_projkps_mode'] = out_projected_joints_reshape
        preds['out_cam_t'] = pred_cam_t.reshape(batch_size, self.num_modes, -1)
        preds['min_mode_gt'] = minofn_mode_ids

        for k, v in preds.items():
            if k in self.loss_weights:
                preds[k] = preds[k] * float(self.loss_weights[k])

        preds['objective'] = self.get_objective(preds)
        return preds

    def get_objective(self,preds):
        loss_names = [ k for k,w in self.loss_weights.items() if k in preds ]
        losses_weighted = [ preds[k] for k,w in self.loss_weights.items() if k in preds and w != 0]
        
        for name, item in zip(loss_names, losses_weighted):
            assert losses_weighted[0].shape[0] == item.shape[0], \
                "LOSS BATCH SIZE: {0}: {1} /= {2}: {3}".format(
                loss_names[0], losses_weighted[0].shape[0],
                name, item.shape
            )

        loss = torch.stack(losses_weighted, dim = 1).sum(dim = -1)
        return loss

    def visualize( self, visdom_env_imgs, trainmode, preds, stats, clear_env=False):    
        # Generate GT data from parameters & scale to mms

        # For proper visualization, this should use the H36M Regressor.
        # Initializes visdom environment
        super(Model, self).visualize(
            visdom_env_imgs, trainmode, preds, stats, clear_env)

        gt_verts, _, _ = self.smpl_mini(
            preds['gt_rotmat'], preds['betas'], run_mini=False)
        gt_verts = gt_verts.data.cpu().numpy() * 1000.0

        # Choose batch size and set image
        batch_size = gt_verts.shape[0]
        idx_image = np.random.choice(range(batch_size))

        # Get the correct mapper for the dataset
        dataset_key = preds['dataset_key'][idx_image]
        joint_mapper_h36m, joint_mapper_gt, _ = \
            self.accuracy_metrics.get_mappers(dataset_key)

        # Unnormalize the input image
        images = preds['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        img = np.transpose(images[idx_image].data.cpu().numpy(), (1, 2, 0))

        # Get and unnormalize the GT keypoint data and -> J14
        gt_keypoints = preds['keypoints'][:, 25:, :].data.cpu().numpy()
        gt_keypoints[:, :, :2] = ((gt_keypoints[:, :, :2] + 1) * 0.5) * global_config.IMG_RES # Unnormalize 2d keypoints

        keypoints_2d = gt_keypoints[idx_image, joint_mapper_gt, :2]
        keypoints_vis = gt_keypoints[idx_image, joint_mapper_gt, 2]

        # Draw GT data on the image
        img_with_gt = draw_skeleton(
            img, keypoints_2d, 
            draw_edges=True, vis=keypoints_vis)
        img_with_gt = (img_with_gt * 255.0).astype(np.uint8)

        # Find and unnormalize the predcited projected keypoints stored from training
        # (B, M, 49, 2)
        out_proj_keypoints = preds['out_projkps_mode'].data.cpu().numpy()
        out_proj_keypoints[:, :, :, :2] = \
            ((out_proj_keypoints[:, :, :, :2] + 1) * 0.5) * global_config.IMG_RES # Unnormalize 2d keypoints
        out_proj_keypoints = out_proj_keypoints[:, :, 25:, :]

        # Generate SMPL for each predicted mode
        out_verts, _, out_pelvis = self.smpl_mini(
            preds['out_fpose_mode'][idx_image], 
            preds['out_shape_mode'][idx_image])
        
        # Project and unnormalize each mode's SMPL to the image 
        camera_center = torch.zeros(self.num_modes, 2).to(
            out_verts.device)
        camera_rotation = torch.eye(3).unsqueeze(0).expand(
            self.num_modes, -1, -1).to(out_verts.device)

        verts_proj = perspective_projection(
            (out_verts + out_pelvis).reshape(self.num_modes, -1, 3),
            rotation=camera_rotation,
            translation=preds['out_cam_t'][idx_image],
            focal_length=self.focal_length,
            camera_center=camera_center).data.cpu().numpy()

        verts_proj = verts_proj / (global_config.IMG_RES / 2.)
        verts_proj[:, :, :2] = ((verts_proj[:, :, :2] + 1) * 0.5) * global_config.IMG_RES # Unnormalize 2d keypoints

        # Scale the predicted SMPL to mms
        out_verts = out_verts.data.cpu().numpy() * 1000.0
                
        # Scale the input 3D joint data to mms -> S24
        gt_joints = preds['pose_3d'].data.cpu().numpy() * 1000.0
        gt_selected_joints = gt_joints[idx_image, :, :3]

        # Find the min_mode data from forward pass  
        min_mode = preds['min_mode_gt'][idx_image].cpu().data.numpy()

        # Find the predicted 3D keypoints data from forward pass (S24)
        out_model_joints_mode = preds['out_joints_mode'][:, :, 25:, :].cpu().detach().numpy() * 1000.0
        out_model_joints_img = out_model_joints_mode.reshape(
            batch_size, self.num_modes, -1, 3)[idx_image, :, :, :3]  
        
        # Get accuracy metrics for each mode
        metrics, dataset_results = self.accuracy_metrics.run_dataset(
            dataset_key,
            preds['out_fpose_mode'][idx_image],
            preds['out_shape_mode'][idx_image],
            #out_pelvis,
            MinofN.expand_modes(preds['gt_rotmat'], self.num_modes)[idx_image],
            MinofN.expand_modes(preds['betas'], self.num_modes)[idx_image],
            MinofN.expand_modes(preds['pose_3d'], self.num_modes)[idx_image],
            MinofN.expand_modes(preds['gender'], self.num_modes)[idx_image])
        
        mpjpe_mode = metrics['mpjpe']
        reco_mode = metrics['r_error']
        vert_mode = metrics['s_error']

        vis_plot = VisdomPlotly(visdom_env_imgs, stats.visdom_server, stats.visdom_port)
        vis_plot.make_fig(
            1, 4, stats.epoch, stats.it[trainmode], idx_image, 
            "{0}[P3D={1},SMPL={2},STATIC={3},DEL={4}]".format(
                global_config.DATASET_LIST[dataset_key],
                preds['has_pose_3d'][idx_image],
                preds['has_smpl'][idx_image],
                preds['has_static'][idx_image],
                global_config.DELETE_TYPE[preds['delete_type'][idx_image]]),
            "[M{0}]: [{1:.2f}, {2:.2f}, {3:.2f}]".format(
                min_mode, 
                mpjpe_mode[min_mode], 
                reco_mode[min_mode],
                vert_mode[min_mode]))

        # Add image and GT joints to first two panels
        vis_plot.add_image(img_with_gt)
        vis_plot.add_2d_points(
            keypoints_2d.reshape(-1, 2), 1, 1, 'Input (Joints)', 'green')
        vis_plot.add_2d_points(
            keypoints_2d.reshape(-1, 2), 1, 2, 'Input (Joints)', 'green')

        # Add GT joints & GT point cloud mesh to 3rd panel
        vis_plot.add_3d_points(
            gt_selected_joints.reshape(-1, 3) * 0.1, 1, 3, 'GT', 'green', 
            visible='legendonly')
        vis_plot.add_3d_points(
            gt_verts[idx_image].reshape(-1, 3) * 0.1, 1, 3, 'GT', 'green', 
            s=1, opacity=0.5, hide_text=True)

        mode_range = list(range(self.num_modes))
        for mode_id in mode_range[::20]: 
            output_model_joints_mode = out_model_joints_img[mode_id]
            out_verts_proj = verts_proj.reshape(self.num_modes, -1, 2)[mode_id, :, :2]            
            out_proj_kps = out_proj_keypoints[idx_image, mode_id, joint_mapper_gt, :2]

            output_verts = out_verts.reshape(self.num_modes, -1, 3)[mode_id, :, :3]
            rmse_proj = np.sqrt(np.sum((out_proj_kps - keypoints_2d) ** 2, axis=-1)).mean()

            visible_3d = 'legendonly'
            if mode_id == min_mode:
                visible_3d = True
                
            # Add pred joints & pred point cloud mesh to 3rd panel
            vis_plot.add_3d_points(
                output_model_joints_mode.reshape(-1, 3) * 0.1, 1, 3, 
                '[M{0}]: 3D Joints=[{1:.2f}, {2:.2f}]'.format(
                    mode_id, mpjpe_mode[mode_id], reco_mode[mode_id]), 
                    'blue', visible=visible_3d)

            vis_plot.add_3d_points(
                output_verts.reshape(-1, 3) * 0.1, 1, 3, 
                '[M{0}]: 3D Mesh={1:.2f}'.format(
                    mode_id, vert_mode[mode_id]), 'red', 
                s=1, opacity=0.5, visible=visible_3d, hide_text=True)

            # Add projected joints and projected mesh to 2nd panel
            vis_plot.add_2d_points(
                out_proj_kps, 1, 2, 
                '[M{0}]: Proj Joints={1:.2f}'.format(mode_id, rmse_proj), 
                'blue', visible=visible_3d)

            vis_plot.add_2d_points(
                out_verts_proj.reshape(-1, 2), 1, 2, 
                '[M{0}]: Proj Mesh'.format(mode_id), 'red', 
                scale=1, opacity=0.5, visible=visible_3d, hide_text=True)

        # Display usage of each mode as a histogram
        vis_plot.add_bar(preds['min_mode_gt'].data.cpu().numpy(), self.num_modes, 1, 4, 'Best Mode (Batch)')
        vis_plot.show()
        
        print ("Updated visdom... Huzzah!")