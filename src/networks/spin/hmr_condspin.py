import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import numpy as np
import math
import config as global_config
from utils.geometry import rot6d_to_rotmat
from utils.minofn import MinofN
from utils.conversions import rot3x3_to_axis_angle

class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HMR_CondSpin(nn.Module):
    """SMPL Iterative Regressor with ResNet50 backbone"""

    def __init__(
        self,
        num_modes,
        num_flows,
        combine_flow_modes,
        block,
        layers,
        smpl_mean_params,
        vae_dim,
    ):
        self.inplanes = 64
        super(HMR_CondSpin, self).__init__()
        self.npose = 23 * 6
        self.nglobal = 1 * 6
        self.nposeaxis = 23 * 3
        self.nshape = 10
        self.ncam = 3
        self.num_modes = num_modes
        self.num_flows = num_flows
        self.combine_flow_modes = combine_flow_modes
        self.vae_dim = vae_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, num_modes=1
        )  # Layer4 is the SPIN prediction
        self.layer5 = self._make_layer(
            block, 512, layers[3], stride=2, num_modes=self.num_modes, inplanes=1024
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)

        output_size = self.nshape + self.npose + self.nglobal + self.ncam
        self.fc1 = nn.Linear(512 * block.expansion + output_size, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        # self.num_modes * (2048 + 3 + 10 + 69)
        fc3_spin_input = (
            512 * block.expansion + self.nposeaxis + self.nshape + self.ncam
        )
        if self.combine_flow_modes:
            fc3_spin_input = fc3_spin_input * (self.num_modes + 1)

        fc3_flow = self.nposeaxis * self.num_flows * 2
        self.fc3 = nn.Linear(fc3_spin_input + fc3_flow, 1024)
        self.drop3 = nn.Dropout()
        self.fc4 = nn.Linear(1024, 1024)
        self.drop4 = nn.Dropout()
        self.fc5 = nn.Linear(512 * block.expansion + self.vae_dim, 1024)
        self.drop5 = nn.Dropout()
        self.fc6 = nn.Linear(1024, 512 * block.expansion)
        self.drop6 = nn.Dropout()

        self.decpose = nn.Linear(1024, self.npose + self.nglobal)
        self.decshape = nn.Linear(1024, self.nshape)
        self.deccam = nn.Linear(1024, self.ncam)

        flow_output_size = self.nposeaxis * num_flows
        self.decflow_mu = nn.Linear(1024, flow_output_size)
        self.decflow_logsigma = nn.Linear(1024, flow_output_size)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decflow_mu.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decflow_logsigma.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params["pose"][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params["shape"][:].astype("float32")
        ).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params["cam"]).unsqueeze(0)

        self.register_buffer("init_pose", init_pose)
        self.register_buffer("init_shape", init_shape)
        self.register_buffer("init_cam", init_cam)

    def _make_layer(self, block, planes, blocks, stride=1, num_modes=1, inplanes=None):
        if inplanes is None:
            inplanes = self.inplanes

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # bjb_edit
        if num_modes > 1:
            out_channels = planes * 4 * num_modes
            layers[-1].conv3 = nn.Conv2d(
                planes, out_channels, kernel_size=1, bias=False
            )
            layers[-1].bn3 = nn.BatchNorm2d(out_channels)
            layers[-1].downsample = lambda x: x.repeat(1, num_modes, 1, 1)

        return nn.Sequential(*layers)

    def encode(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # SPIN prediction
        x5 = self.layer5(x3)  # Multi-modes: Needs to be unfrozen for M>1

        xf_spin = self.avgpool(x4).view(batch_size, 1, -1)
        xf_extra = self.avgpool(x5).view(batch_size, self.num_modes, -1)

        return xf_spin, xf_extra

    def decode_smpl(
        self, xfm, init_pose=None, init_shape=None, init_cam=None, n_iter=3
    ):
        batch_size = xfm.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        num_modes = xfm.shape[1]
        init_pose = MinofN.repeat_modes(init_pose, num_modes)  # (B, M, 144)
        init_cam = MinofN.repeat_modes(init_cam, num_modes)  # (B, M, 3)
        init_shape = MinofN.repeat_modes(init_shape, num_modes)  # (B, M, 3)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for m in range(num_modes):
            for i in range(n_iter):
                xc = torch.cat(
                    [xfm[:, m], pred_pose[:, m], pred_shape[:, m], pred_cam[:, m]],
                    dim=1,
                )

                # Note: Dropout makes the initial predictions slightly different
                xc = self.fc1(xc)
                xc = self.drop1(xc)
                xc = self.fc2(xc)
                xc = self.drop2(xc)

                pred_pose[:, m] = self.decpose(xc) + pred_pose[:, m]
                pred_shape[:, m] = self.decshape(xc) + pred_shape[:, m]
                pred_cam[:, m] = self.deccam(xc) + pred_cam[:, m]

        pred_pose = pred_pose.reshape(batch_size, num_modes, 24, 6)
        pred_shape = pred_shape.reshape(batch_size, num_modes, self.nshape)
        pred_cam = pred_cam.reshape(batch_size, num_modes, self.ncam)

        # convert SPIN rotmats -> axisangle
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, num_modes, 24, 3, 3)
        axis_angle = rot3x3_to_axis_angle(pred_rotmat).reshape(
            batch_size, num_modes, -1
        )

        spin_results = {
            "pred_rotmat": pred_rotmat,
            "axis_angle": axis_angle,
            "pred_shape": pred_shape,
            "pred_camera": pred_cam,
        }

        return spin_results

    def decode_flow(self, xfm, pred_axis, pred_shape, pred_cam, n_iter=3):
        batch_size = xfm.shape[0]
        spin_data = torch.cat([xfm, pred_axis, pred_shape, pred_cam], dim=-1)

        pred_flow_mu = torch.zeros(batch_size, self.nposeaxis * self.num_flows).to(
            pred_axis.device
        )
        pred_flow_logsigma = torch.zeros(
            batch_size, self.nposeaxis * self.num_flows
        ).to(pred_axis.device)

        for i in range(n_iter):
            xf_modes = torch.cat(
                [
                    spin_data.reshape(batch_size, -1),
                    pred_flow_mu,
                    pred_flow_logsigma,
                ],
                dim=1,
            )

            xf_modes = self.fc3(xf_modes)
            xf_modes = self.drop3(xf_modes)
            xf_modes = self.fc4(xf_modes)
            xf_modes = self.drop4(xf_modes)

            # Predict mu + std dev for the flow
            pred_flow_mu = self.decflow_mu(xf_modes)
            pred_flow_logsigma = self.decflow_logsigma(xf_modes)

        pred_flow_mu = pred_flow_mu.reshape(batch_size, self.num_flows, -1)
        pred_flow_logsigma = pred_flow_logsigma.reshape(
            batch_size, self.num_flows, -1
        )

        flow_results = {"flow_mu": pred_flow_mu, "flow_logsigma": pred_flow_logsigma}

        return flow_results

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        xf_spin, xf_extra = self.encode(x)
        xfm = torch.cat([xf_spin, xf_extra], dim=1)
        print("Has NaN: {0}".format(~(~torch.isnan(xfm)).all()))
        hmr_results = self.decode_smpl(xfm, n_iter=n_iter)
        hmr_results.update(
            self.decode_flow(
                xfm,
                hmr_results["axis_angle"][
                    :, :, 3:
                ],
                hmr_results["pred_shape"],
                hmr_results["pred_camera"],
            )
        )
        hmr_results.update(
            {
                "spin_global_rotmat": hmr_results["pred_rotmat"][:, 0, [0]],
                "spin_local_rotmat": hmr_results["pred_rotmat"][:, 0, 1:],
                "spin_global_axis": hmr_results["axis_angle"][:, 0, :3],
                "spin_local_axis": hmr_results["axis_angle"][:, 0, 3:],
                "spin_shape": hmr_results["pred_shape"][:, 0],
                "spin_camera": hmr_results["pred_camera"][:, 0],
                "pred_global_rotmat": hmr_results["pred_rotmat"][:, 1:, [0]],
                "pred_local_rotmat": hmr_results["pred_rotmat"][:, 1:, 1:],
                "pred_global_axis": hmr_results["axis_angle"][:, 1:, :3],
                "pred_local_axis": hmr_results["axis_angle"][:, 1:, 3:],
                "pred_shape": hmr_results["pred_shape"][:, 1:],
                "pred_camera": hmr_results["pred_camera"][:, 1:],
            }
        )

        return hmr_results

class ConditionalHMR(nn.Module):
    def __init__(
        self,
        num_modes,
        num_flows,
        combine_flow_modes=True,
        pretrained="spin",
        freeze_weights=True,
        initialize_decpose=True,
        layer5_noise=0.0,
        vae_dim=64,
        **kwargs
    ):

        super(ConditionalHMR, self).__init__()

        smpl_mean_params = global_config.SMPL_MEAN_PARAMS
        self.model = HMR_CondSpin(
            num_modes,
            num_flows,
            combine_flow_modes,
            Bottleneck,
            [3, 4, 6, 3],
            smpl_mean_params,
            vae_dim,
            **kwargs
        )

        if pretrained == "imagenet":
            resnet_imagenet = resnet.resnet50(pretrained=True)
            self.model.load_state_dict(resnet_imagenet.state_dict(), strict=False)

        weights_to_freeze = {}
        if pretrained == "spin":
            modewise_layers = [
                "layer4.2.conv3.weight",
                "layer4.2.bn3.weight",
                "layer4.2.bn3.bias",
                "layer4.2.bn3.running_mean",
                "layer4.2.bn3.running_var",
            ]

            hmr_checkpoint = torch.load(global_config.HMR_PRETRAINED)["model"]
            own_state = self.model.state_dict()
            for name, param in hmr_checkpoint.items():
                # If the flow is on, keep the decpose vector as 'unlearnt'
                # and not frozen
                if initialize_decpose == False:
                    if "decpose" in name:
                        print("xxx Not initializing decpose")
                        continue

                if name not in own_state:
                    print("xxx Rejecting param {0} due to nonexistance".format(name))
                    continue
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data

                # Copy standard layers and optionally freeze
                if own_state[name].shape == param.shape:
                    # <Turn on for debugging>
                    # print ("<-- LOAD param {0}".format(name))
                    own_state[name].copy_(param)
                    weights_to_freeze[name] = param
                else:
                    print("xxx Rejecting param {0} due to shape mismatch".format(name))

                if "layer4" in name:
                    layer5_name = name.replace("layer4", "layer5")
                    if name in modewise_layers:
                        rem_ones = [1] * len(param.shape[1:])
                        param_exp = param.repeat(num_modes, *rem_ones)

                        # Add noise to prevent initialization to identical modes
                        additive_noise = torch.randn_like(param_exp) * layer5_noise
                        param_exp = param_exp + additive_noise

                        print(
                            "<-- LOAD [EXPANDED] {0} into {1} [{2} into {3}] with noise {4}".format(
                                name,
                                layer5_name,
                                param_exp.shape,
                                own_state[layer5_name].shape,
                                layer5_noise,
                            )
                        )

                        own_state[layer5_name].copy_(param_exp)
                    else:
                        print("<-- LOAD param {0} into {1}".format(name, layer5_name))
                        own_state[layer5_name].copy_(param)
                        weights_to_freeze[layer5_name] = param

            if freeze_weights:
                for name, param in self.model.named_parameters():
                    if name in weights_to_freeze:
                        param.requires_grad = False

        print("HMR Checkpoint loaded")

    def forward(self, *args):
        return self.model(*args)