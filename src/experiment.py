import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append("../")

from exp_manager.config import (
    get_config_from_file,
    get_arg_parser,
    set_config,
)
from exp_manager.utils import get_net_input, pprint_dict
from exp_manager.vis_utils import get_visdom_env
from exp_manager.cache_preds import cache_preds
from exp_manager.stats import Stats
from exp_manager.base_exp import BaseExperiment

from exp_manager.model_io import (
    find_last_checkpoint,
    purge_epoch,
    load_model,
    get_checkpoint,
    save_model,
    load_stats,
)

from model import Model
from optimizer import Optimizer
from experiment_config import ExperimentConfig
from datetime import datetime

import config

from datasets.dataset_zoo import dataset_zoo
from eval_zoo import eval_zoo
import torch
import torch.nn as nn
import time
import numpy as np
import copy
import ast
import pandas as pd


class Experiment(BaseExperiment):
    def __init__(self, exp_config, model, optimizer, dataset_zoo, eval_zoo):

        self.exp_config = exp_config
        self.model = model
        self.optimizer = optimizer
        self.dataset_zoo = dataset_zoo
        self.eval_zoo = eval_zoo

        exp = self.exp_config()

        parser = get_arg_parser(type(exp))
        parsed = parser.parse_args()
        # If the path exists, then set these as default
        # Let command line args override!
        cfg_file = os.path.join(parsed.exp_dir, "expconfig.yaml")
        if os.path.exists(cfg_file):
            cfg_load = get_config_from_file(cfg_file)
            print("<- Loaded base config settings from: {0}".format(cfg_file))
            parser = get_arg_parser(type(exp), default=cfg_load)
            parsed = parser.parse_args()

        set_config(exp.cfg, vars(parsed))
        pprint_dict(exp.cfg)
        self.cfg = exp.cfg

        if self.cfg.test_batch_size is None:
            self.cfg.test_batch_size = self.cfg.batch_size

        total_gpus = torch.cuda.device_count()
        if self.cfg.gpu_idx is None:
            self.cfg.gpu_idx = list(range(total_gpus))

        if type(self.cfg.gpu_idx) is str:
            self.cfg.gpu_idx = ast.literal_eval(self.cfg.gpu_idx)

        num_gpus = len(self.cfg.gpu_idx)
        print("Let's use {0} of available {1} GPUs!".format(num_gpus, total_gpus))

        print("-> Batch Size: {0}".format(self.cfg.batch_size))
        print("-> LR: {0}".format(self.cfg.SOLVER.lr))

    def run_training(self):
        datasets, model, optimizer, scheduler, stats, loaded_path = self.initialize(
            self.cfg
        )

        # Save the initialization
        # self.store_checkpoints(model, stats, optimizer, 999, self.cfg)

        start_epoch = scheduler.last_epoch
        for epoch in range(start_epoch, self.cfg.SOLVER.max_epochs):
            self.run_epoch(
                datasets, model, optimizer, scheduler, epoch, stats, self.cfg
            )

    def run_evaluation(self):
        datasets, model, optimizer, scheduler, stats, loaded_path = self.initialize(
            self.cfg
        )

        _, _, testloader = datasets

        start_epoch = scheduler.last_epoch
        if testloader is not None:
            self.run_eval(self.cfg, model, stats, testloader, update=False)

        self.store_checkpoints(model, stats, optimizer, start_epoch, self.cfg)

    def init_model(self, cfg, add_log_vars=None):

        # get the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.model(**cfg.MODEL)

        if hasattr(model, "log_vars"):
            log_vars = copy.deepcopy(model.log_vars)
        else:
            log_vars = ["objective"]
        if add_log_vars is not None:
            log_vars.extend(copy.deepcopy(add_log_vars))

        model = nn.DataParallel(model, device_ids=cfg.gpu_idx)
        model = model.to(device)

        # obtain the network outputs that should be logged
        visdom_env_charts = get_visdom_env(cfg) + "_charts"

        # init stats struct
        stats = Stats(
            log_vars,
            visdom_env=visdom_env_charts,
            verbose=False,
            visdom_server=cfg.visdom_server,
            visdom_port=cfg.visdom_port,
        )

        return model, stats, log_vars, visdom_env_charts

    def run_epoch(self, datasets, model, optimizer, scheduler, epoch, stats, cfg):

        trainloader, valloader, testloader = datasets

        with stats:  # automatic new_epoch and plotting of stats at every epoch start
            print("scheduler lr = {0:.2e}".format(float(scheduler.get_lr()[-1])))

            # # train loop
            self.trainvalidate(
                model,
                stats,
                epoch,
                trainloader,
                optimizer,
                False,
                visdom_env_root=get_visdom_env(cfg),
                **cfg
            )

            # eval loop (optional)
            if epoch % self.cfg.DATASET.length_divider == 0:
                if testloader is not None:
                    self.run_eval(cfg, model, stats, testloader)
            else:
                for t_key, t_val in stats.stats["test"].items():
                    for _ in range(int(self.cfg.DATASET.length_divider) - 1):
                        stats.stats["test"][t_key].update(t_val.history[-1])
                stats.print(stat_set="test")
                print("Completed copying previous tests")

            assert stats.epoch == epoch, "inconsistent stats!"

            if cfg.store_checkpoints_purge > 0:
                self.purge_checkpoint(epoch, cfg)

            # save model
            if cfg.store_checkpoints:
                self.store_checkpoints(model, stats, optimizer, epoch, cfg)

            scheduler.step()

    def init_datasets(self, cfg):
        # setup datasets
        dset_train, dset_val, dset_test = self.dataset_zoo(**cfg.DATASET)

        # init loaders
        if dset_train is not None:
            trainloader = torch.utils.data.DataLoader(
                dset_train,
                num_workers=cfg.num_workers,
                pin_memory=True,
                batch_size=cfg.batch_size,
                shuffle=True,
            )
        else:
            trainloader = None

        if dset_val is not None:
            valloader = torch.utils.data.DataLoader(
                dset_val,
                num_workers=cfg.num_workers,
                pin_memory=True,
                batch_size=cfg.test_batch_size,
                shuffle=True,
            )
        else:
            valloader = None

        # test loaders
        if dset_test is not None:
            testloader = {}
            eval_vars = []
            for test_name, test_dset in dset_test.items():
                testloader[test_name] = torch.utils.data.DataLoader(
                    test_dset,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    batch_size=cfg.test_batch_size,
                    shuffle=True,
                )
                _, _, eval_vars_dset = self.eval_zoo(test_dset)
                eval_vars += eval_vars_dset
        else:
            testloader = None
            eval_vars = None

        return (trainloader, valloader, testloader), eval_vars

    def trainvalidate(
        self,
        model,
        stats,
        epoch,
        loader,
        optimizer,
        validation,
        bp_var="objective",
        metric_print_interval=5,
        visualize_interval=100,
        visdom_env_root="trainvalidate",
        **kwargs
    ):

        if validation:
            model.eval()
            trainmode = "val"
        else:
            model.train()
            trainmode = "train"

        t_start = time.time()

        # clear the visualisations on the first run in the epoch
        clear_visualisations = True

        # get the visdom env name
        visdom_env_imgs = visdom_env_root + "_images_" + trainmode

        n_batches = len(loader)
        for it, batch in enumerate(loader):
            last_iter = it == n_batches - 1

            # move to gpu where possible
            net_input = get_net_input(batch)

            # the forward pass
            if not validation:
                preds = model(epoch, net_input)

                # make sure we dont overwrite something
                assert not any(k in preds for k in net_input.keys())
                preds.update(net_input)  # merge everything into one big dict

                optimizer.zero_grad()
                gen_loss = preds["objective"].mean()  # mean over the batch
                gen_loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    preds = model(epoch, net_input)

                # make sure we dont overwrite something
                assert not any(k in preds for k in net_input.keys())
                preds.update(net_input)  # merge everything into one big dict

            # update the stats logger
            stats.update(preds, time_start=t_start, stat_set=trainmode)
            assert stats.it[trainmode] == it, "inconsistent stat iteration number!"

            # print textual status update
            if (it % metric_print_interval) == 0 or last_iter:
                stats.print(stat_set=trainmode, max_it=n_batches)

            # visualize results
            if (visualize_interval > 0) and (it % visualize_interval) == 0:
                with torch.no_grad():
                    model.module.visualize(
                        visdom_env_imgs,
                        trainmode,
                        preds,
                        stats,
                        clear_env=clear_visualisations,
                    )
                # clear_visualisations = False

    def restore_model(
        self,
        cfg,
        model,
        stats,
        log_vars,
        visdom_env_charts,
        force_load=False,
        clear_stats=False,
    ):
        # find the last checkpoint
        if cfg.resume_epoch > 0:
            model_path = get_checkpoint(cfg.exp_dir, cfg.resume_epoch)
        else:
            model_path = find_last_checkpoint(cfg.exp_dir)

        if cfg.resume_checkpoint is not None:
            model_path = cfg.resume_checkpoint

        optimizer_state = None
        loaded_path = None

        print("<- Attempting to load: {0}".format(model_path))

        if model_path is not None:
            print("found previous model %s" % model_path)
            if force_load or cfg.resume:
                print("   -> resuming")
                model_state_dict, stats_load, optimizer_state = load_model(model_path)
                if not clear_stats:
                    stats = stats_load
                else:
                    print("   -> clearing stats")

                own_state = model.state_dict()
                for name, param in model_state_dict.items():
                    try:
                        own_state[name].copy_(param)
                    except:
                        print("Unable to load: {0}".format(name))                       
                        

                # model.load_state_dict(model_state_dict,strict=True)
                model.log_vars = log_vars
                loaded_path = model_path
            else:
                print("   -> but not resuming -> starting from scratch")

        if cfg.MODEL.init_flow != "":
            flow_state_dict = torch.load(cfg.MODEL.init_flow)
            print (f"<- Loading model_realnvp from: {cfg.MODEL.init_flow}")
            own_state = model.state_dict()
            for name, param in flow_state_dict.items():
                if name.split(".")[1] == "model_realnvp":
                    if name not in own_state:
                        continue
                    if isinstance(param, torch.nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    own_state[name].copy_(param)
            

        # update in case it got lost during load:
        stats.visdom_env = visdom_env_charts
        stats.visdom_server = cfg.visdom_server
        stats.visdom_port = cfg.visdom_port
        stats.plot_file = os.path.join(cfg.exp_dir, "train_stats.pdf")

        return model, optimizer_state, stats, loaded_path

    def pprint_results(self, results, ambiguous, dump_dir="results"):
        result_lst = [[k.replace("r_error", "reco"), float(v)] for k, v in results.items() ]
        df = pd.DataFrame(result_lst)
        df[['EVAL', 'Dataset', 'Mode', 'Metric']] = df[0].str.split("_", expand=True)
        df = df.drop(labels=["EVAL"], axis=1)
        df[["Weighted", "Mode_ID"]] = df["Mode"].str.split("M", expand=True)
        df["Weighted"] = df["Weighted"].apply(lambda x: x == "W")

        df_todup = df[(~df["Weighted"]) & (df["Mode_ID"] == "01")].copy()
        df_todup["Weighted"] = True

        df = pd.concat([df_todup, df])

        dsets = sorted(df["Dataset"].unique())

        out_dir = os.path.join(dump_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(out_dir)

        for dset in dsets:
            df_det = df[df["Dataset"] == dset]
            df_det = df_det.sort_values(by=["Weighted", "Mode_ID", "Metric"])

            def prep_df(sub_df, name):
                sub_df = sub_df[["Mode_ID", "Metric", 1]].transpose()
                sub_df.columns = sub_df.iloc[0] + "_" + sub_df.iloc[1]
                sub_df = sub_df.drop(["Mode_ID", "Metric"])
                sub_df = sub_df.set_index(pd.Index([name]))
                return sub_df

            amb_pref = "a" if ambiguous else ""
            w_df = prep_df(df_det[df_det["Weighted"]], f"{amb_pref}{dset}_WEIGHT")
            nw_df = prep_df(df_det[~df_det["Weighted"]], f"{amb_pref}{dset}")

            comb_df = pd.concat([w_df, nw_df])

            print (comb_df.to_markdown())
            with open(os.path.join(out_dir, f"{dset}.md"), 'w') as f:
                f.write(comb_df.to_markdown())

    def run_eval(self, cfg, model, stats, loader, update=True):
        results = {}
        for dset_name, dset_loader in loader.items():
            eval_script, cache_vars, eval_vars = self.eval_zoo(dset_loader.dataset)
            cached_preds = cache_preds(model, dset_loader, cache_vars=cache_vars)
            results_dset, _ = eval_script(cached_preds, eval_vars=eval_vars)
            results.update(results_dset)
            print("-> Completed test: {0}".format(dset_name))
            print(results)
            self.pprint_results(results, dset_loader.dataset.ambiguous)

        if update:
            stats.update(results, stat_set="test")
            stats.print(stat_set="test")

        print("Completed all tests")


if __name__ == "__main__":
    experiment = Experiment(ExperimentConfig, Model, Optimizer, dataset_zoo, eval_zoo)

    if experiment.cfg.evaluate:
        experiment.run_evaluation()
    else:
        experiment.run_training()
