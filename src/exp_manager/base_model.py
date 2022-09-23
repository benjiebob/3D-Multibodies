import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle as pkl
import time
import os

from .utils import auto_init_args
from .vis_utils import get_visdom_connection

class BaseModel(torch.nn.Module):
    def __init__( self, 
            loss_weights={
              'loss_LOSS1': 1.0
            },
            log_vars=[
                'objective', 'loss_LOSS1', 'acc_METRIC'
            ], 
            **kwargs):

        super(BaseModel, self).__init__()

        # autoassign constructor params to self
        auto_init_args(self)
        self.loss_weights = loss_weights

    def forward(self, batch_input: dict) -> dict:
        preds = {}
        return preds

    def visualize(self, visdom_env_imgs, trainmode, preds, stats, clear_env=False) -> None:
        self.viz = get_visdom_connection(server=stats.visdom_server,port=stats.visdom_port)
        if not self.viz.check_connection():
            print("no visdom server! -> skipping batch vis")
            return

        if clear_env: # clear visualisations
            print("  ... clearing visdom environment")
            self.viz.close(env=visdom_env_imgs,win=None)