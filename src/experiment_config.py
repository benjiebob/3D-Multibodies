import sys

sys.path.append("../")

from exp_manager.config import get_default_args, set_config_from_file
from exp_manager.attr_dict import nested_attr_dict
from exp_manager.utils import auto_init_args
import os

from datasets.dataset_zoo import dataset_zoo
from optimizer import Optimizer
from model import Model


class ExperimentConfig(object):
    def __init__(
        self,
        cfg_file=None,
        model_zoo="./data/torch_zoo/",
        exp_name="",
        exp_idx=0,
        exp_dir=".",
        gpu_idx=None,
        resume=True,
        seed=0,
        resume_epoch=-1,
        resume_checkpoint=None,
        store_checkpoints=True,
        store_checkpoints_purge=3,
        batch_size=48,
        test_batch_size=200,
        num_workers=0,
        eval_name="loss_nll",
        eval_higher_better=False,
        eval_optset="test",
        requeue_epoch_freq=-1,
        visdom_env="",
        visdom_server="http://localhost",
        visdom_port=8097,
        metric_print_interval=5,
        visualize_interval=1000,
        evaluate=False,
        MODEL=get_default_args(Model),
        SOLVER=get_default_args(Optimizer),
        DATASET=get_default_args(dataset_zoo),
    ):

        self.cfg = get_default_args(ExperimentConfig)
        if cfg_file is not None:
            set_config_from_file(self.cfg, cfg_file)
        else:
            auto_init_args(self, tgt="cfg", can_overwrite=True)
        self.cfg = nested_attr_dict(self.cfg)
