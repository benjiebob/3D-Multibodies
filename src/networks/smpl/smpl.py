"""
This file contains the definition of the SMPL model
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

import config

try:
    import cPickle as pickle
except ImportError:
    import pickle

from .geometric_layers import rodrigues
from utils import constants

from .vertex_joint_selector import VertexJointSelector
from .vertex_ids import vertex_ids as VERTEX_IDS
import os


class SMPL(nn.Module):
    def __init__(self, model_file):
        super(SMPL, self).__init__()
        with open(model_file, "rb") as f:
            smpl_model = pickle.load(f, encoding="latin1")

        # This is the smpl_model joint regressor, converted to numpy from chumpy
        mb3d_regressor= np.load(config.JOINT_REGRESSOR_3DMB)["J_regressor"]
        J_regressor = torch.from_numpy(mb3d_regressor).float()
        self.register_buffer("J_regressor", J_regressor.contiguous())
        self.register_buffer("weights", torch.FloatTensor(smpl_model["weights"]))
        self.register_buffer("posedirs", torch.FloatTensor(smpl_model["posedirs"]))
        self.register_buffer("v_template", torch.FloatTensor(smpl_model["v_template"]))
        self.register_buffer(
            "shapedirs", torch.FloatTensor(np.array(smpl_model["shapedirs"]))
        )
        self.register_buffer(
            "faces", torch.from_numpy(smpl_model["f"].astype(np.int64))
        )
        self.register_buffer(
            "kintree_table",
            torch.from_numpy(smpl_model["kintree_table"].astype(np.int64)),
        )
        id_to_col = {
            self.kintree_table[1, i].item(): i
            for i in range(self.kintree_table.shape[1])
        }
        self.register_buffer(
            "parent",
            torch.LongTensor(
                [
                    id_to_col[self.kintree_table[0, it].item()]
                    for it in range(1, self.kintree_table.shape[1])
                ]
            ),
        )

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        # SMPL and SMPL-H share the same topology, so any extra joints can
        # be drawn from the same place
        vertex_ids = VERTEX_IDS["smplh"]
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids)

        spin_ids = self.vertex_joint_selector.extra_joints_idxs
        J_spin_extra = torch.zeros(len(spin_ids), self.v_template.shape[0])
        J_spin_extra[range(len(spin_ids)), spin_ids] = 1.0
        self.register_buffer("J_regressor_spin", J_spin_extra)

        JOINT_REGRESSOR_TRAIN_EXTRA = config.JOINT_REGRESSOR_TRAIN_EXTRA
        J_regressor_extra = torch.from_numpy(
            np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        ).float()
        self.register_buffer("J_regressor_extra", J_regressor_extra)

        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        self.joint_map = torch.tensor(joints, dtype=torch.long)

        J_regressor_h36m = torch.from_numpy(
            np.load(config.JOINT_REGRESSOR_H36M)
        ).float()
        self.register_buffer("J_regressor_h36m", J_regressor_h36m)

        mask_Jregressor = (J_regressor > 1e-9).any(dim=0)
        mask_Jregressor_spin = (J_spin_extra > 1e-9).any(dim=0)
        mask_Jregressor_extra = (J_regressor_extra > 1e-9).any(dim=0)
        mask = mask_Jregressor | mask_Jregressor_extra | mask_Jregressor_spin

        self.register_buffer("mask", mask.float())  # hack for torch distributed

    def forward(self, pose, beta, run_mini=False, use_h36m_regressor=False):
        device = pose.device
        batch_size = pose.shape[0]

        if run_mini:
            mask = self.mask.bool()  # hack for torch distributed
        else:
            mask = torch.ones_like(self.mask).bool()  # hack for torch distributed

        v_template = self.v_template[mask][None, :]
        shapedirs = (
            self.shapedirs[mask].view(-1, 10)[None, :].expand(batch_size, -1, -1)
        )

        beta = beta[:, :, None]
        v_shaped = (
            torch.matmul(shapedirs, beta).view(-1, v_template.shape[1], 3) + v_template
        )  # This is the same

        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor[:, mask], v_shaped[i]))
        J = torch.stack(J, dim=0)  # Error ~ 1e-5

        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)

        posedirs = self.posedirs[mask].view(-1, 207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(
            -1, v_template.shape[1], 3
        )  # Identically equal

        # Below is all joint processing
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = (
            torch.FloatTensor([0, 0, 0, 1])
            .to(device)
            .view(1, 1, 1, 4)
            .expand(batch_size, 24, -1, -1)
        )
        G_ = torch.cat([G_, pad_row], dim=2)
        G = G_.clone()
        for i in range(1, 24):
            G[:, i, :, :] = torch.matmul(G[:, self.parent[i - 1], :, :], G_[:, i, :, :])

        joints_smpl24 = G[:, :, :3, -1].clone()  # Error ~ 1e-5

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(
            batch_size, 24, 4, 1
        )
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest

        T = (
            torch.matmul(
                self.weights[mask], G.permute(1, 0, 2, 3).contiguous().view(24, -1)
            )
            .view(v_template.shape[1], batch_size, 4, 4)
            .transpose(0, 1)
        )  # Error 1e-6
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1
        )  # Error = 0
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]  # 1e-5 error

        joints = self.get_joints(v, mask, use_h36m_regressor=use_h36m_regressor)

        joints_copy = joints.clone()
        pelvis = (joints_copy[:, [27], :] + joints_copy[:, [28], :]) * 0.5
        joints = joints - pelvis
        v = v - pelvis

        return v, joints, pelvis

    def get_joints(self, vertices, mask, use_h36m_regressor=False):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """

        joints = torch.einsum("bik,ji->bjk", [vertices, self.J_regressor[:, mask]])
        joints_spin = torch.einsum(
            "bik,ji->bjk", [vertices, self.J_regressor_spin[:, mask]]
        )
        joints_extra = torch.einsum(
            "bik,ji->bjk", [vertices, self.J_regressor_extra[:, mask]]
        )

        joints = torch.cat((joints, joints_spin, joints_extra), dim=1)
        joints = joints[:, self.joint_map]

        return joints
