import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, time, copy

# torch imports
import numpy as np
import torch
import itertools

from .collate import default_collate
from .config import set_config_from_file, set_config, \
                    get_arg_parser, dump_config, get_default_args

from .attr_dict import nested_attr_dict
from .utils import auto_init_args, get_net_input, pprint_dict
from .stats import Stats
from .vis_utils import get_visdom_env
from .model_io import \
    find_last_checkpoint, purge_epoch, \
    load_model, get_checkpoint, save_model, \
    load_stats

from .cache_preds import cache_preds
from .vis_utils import get_visdom_connection, denorm_image_trivial

import torch.nn as nn
import pickle as pkl

class BaseExperiment():
    def __init__(self, exp_config, model, optimizer, dataset_zoo, eval_zoo):

        self.exp_config = exp_config
        self.model = model
        self.optimizer = optimizer
        self.dataset_zoo = dataset_zoo
        self.eval_zoo = eval_zoo

        exp = self.exp_config()
        parser = get_arg_parser(type(exp))
        parsed = parser.parse_args()
        set_config(exp.cfg,vars(parsed))

        pprint_dict(exp.cfg)

        self.cfg = exp.cfg

        num_gpus = torch.cuda.device_count()
        print("Let's use", num_gpus, "GPUs!")

        # assert torch.cuda.device_count() == 1, "Currently only 1 GPU is supported"

    def restore_model(self, cfg, model, stats, log_vars, visdom_env_charts, force_load=False,clear_stats=False):
         # find the last checkpoint
        if cfg.resume_epoch > 0:
            model_path = get_checkpoint(cfg.exp_dir,cfg.resume_epoch)
        else:
            model_path = find_last_checkpoint(cfg.exp_dir)

        if cfg.resume_checkpoint is not None:
            model_path = cfg.resume_checkpoint
        
        optimizer_state = None
        loaded_path = None

        print ("<- Attempting to load: {0}".format(model_path))

        if model_path is not None:
            print( "found previous model %s" % model_path )
            if force_load or cfg.resume:
                print( "   -> resuming" )
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
                        print ("Unable to load: {0}".format(name))

                # model.load_state_dict(model_state_dict,strict=True)
                model.log_vars = log_vars
                loaded_path = model_path
            else:
                print( "   -> but not resuming -> starting from scratch" )
        
        # update in case it got lost during load:
        stats.visdom_env    = visdom_env_charts
        stats.visdom_server = cfg.visdom_server
        stats.visdom_port   = cfg.visdom_port
        stats.plot_file = os.path.join(cfg.exp_dir,'train_stats.pdf')
        stats.synchronize_logged_vars(log_vars)

        # TODO: Make this fix
        # With resume checkpoint = True, don't initialize the optimizer
        # if cfg.resume_checkpoint is not None:
        #     optimizer_state = None

        return model, optimizer_state, stats, loaded_path

    def init_model(
        self,cfg,add_log_vars=None):

        # get the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.model(**cfg.MODEL)
        
        if hasattr(model,'log_vars'):
            log_vars = copy.deepcopy(model.log_vars)
        else:
            log_vars = ['objective']
        if add_log_vars is not None:
            log_vars.extend(copy.deepcopy(add_log_vars))

        model = nn.DataParallel(model)
        model = model.to(device)

        # obtain the network outputs that should be logged    
        visdom_env_charts = get_visdom_env(cfg) + "_charts"

        # init stats struct
        stats = Stats( log_vars, visdom_env=visdom_env_charts, \
                    verbose=False, visdom_server=cfg.visdom_server, \
                    visdom_port=cfg.visdom_port )    

        return model, stats, log_vars, visdom_env_charts

    def purge_checkpoint(self, epoch, cfg):
        # delete previous models if required
        for prev_epoch in range(epoch-cfg.store_checkpoints_purge):
            if prev_epoch % 25 > 0: # keep occasional epochs
                purge_epoch(cfg.exp_dir,prev_epoch)

    def store_checkpoints(self, model, stats, optimizer, epoch, cfg):
        outfile = get_checkpoint(cfg.exp_dir,epoch)
        save_model(model,stats,outfile,optimizer=optimizer)
        eval_val = stats.stats[cfg.eval_optset][cfg.eval_name].val
        if (cfg.eval_higher_better and eval_val > self.best_acc) or \
            (not cfg.eval_higher_better and eval_val < self.best_acc):
                self.best_acc = eval_val
                save_model(model,stats,os.path.join(cfg.exp_dir, 'model_best.pth'), optimizer=optimizer)

    def run_epoch(self, 
        datasets,
        model, optimizer, scheduler, 
        epoch, stats, cfg):

        trainloader, valloader, testloader = datasets

        with stats: # automatic new_epoch and plotting of stats at every epoch start
            print ("scheduler lr = {0:.2e}".format(
                float(scheduler.get_lr()[-1])))

            # train loop
            self.trainvalidate(
                model, stats, epoch, trainloader, optimizer, False, \
                    visdom_env_root=get_visdom_env(cfg), **cfg )
            
            # val loop
            self.trainvalidate(
                model, stats, epoch, valloader,   optimizer, True,  \
                    visdom_env_root=get_visdom_env(cfg), **cfg  )
            
            # eval loop (optional)
            # if testloader is not None:
            #     run_eval(cfg,model,stats,testloader)
            
            assert stats.epoch==epoch, "inconsistent stats!"

            if cfg.store_checkpoints_purge > 0:
                self.purge_checkpoint(epoch, cfg)

            # save model
            if cfg.store_checkpoints:
                self.store_checkpoints(
                    model, stats, optimizer, epoch, cfg)
                
            scheduler.step()

    def init_datasets(self, cfg):
        # setup datasets
        dset_train, dset_val, dset_test = self.dataset_zoo(**cfg.DATASET)

        # init loaders
        trainloader = torch.utils.data.DataLoader(dset_train, 
                            num_workers=cfg.num_workers, pin_memory=True,
                            batch_size=cfg.batch_size, shuffle=True)

        if dset_val is not None:
            valloader = torch.utils.data.DataLoader(dset_val, 
                                num_workers=cfg.num_workers, pin_memory=True,
                                batch_size=cfg.batch_size, shuffle=True)
        else:
            valloader = None

        # test loaders
        if dset_test is not None:
            testloader = torch.utils.data.DataLoader(dset_test, 
                    num_workers=cfg.num_workers, pin_memory=True,
                    batch_size=cfg.batch_size, shuffle=True)
            _,_,eval_vars = self.eval_zoo(cfg.DATASET.dataset_name)
        else:
            testloader = None
            eval_vars = None

        return (trainloader, valloader, testloader), eval_vars

    def init_optimizer(self, model, optimizer_state, cfg):
        opt_init = self.optimizer(
            model,optimizer_state,**cfg.SOLVER)
        return opt_init.optimizer, opt_init.scheduler

    def initialize(self, cfg):
        # run the training loops
        
        # make the exp dir
        os.makedirs(cfg.exp_dir,exist_ok=True)

        # set the seed
        np.random.seed(cfg.seed)

        # dump the exp config to the exp dir
        if not cfg.evaluate:
            dump_config(cfg)

        datasets, eval_vars = self.init_datasets(cfg)
        
        # init the model    
        model, stats, log_vars, visdom_env_charts = self.init_model(
            cfg, add_log_vars=eval_vars)

        model, optimizer_state, stats, loaded_path = self.restore_model(
            cfg, model, stats, log_vars, visdom_env_charts)

        start_epoch = stats.epoch + 1
        
        # init the optimizer
        optimizer, scheduler = self.init_optimizer(model, optimizer_state, cfg)

        print('-------\nloss_weights:')
        for k,w in model.module.loss_weights.items():
            print('%20s: %1.2e' % (k,w) )
        print('-------')

        eval_name = cfg.eval_name
        if cfg.eval_higher_better:
            self.best_acc = 0.0
        else:
            self.best_acc = np.inf

        best_stats_path = os.path.join(cfg.exp_dir, 'model_best_stats.pkl')
        if os.path.exists(best_stats_path):
            best_stats = load_stats(best_stats_path)
            if cfg.eval_optset in best_stats.stats:
                best_val = best_stats.stats[cfg.eval_optset][eval_name].val
                print ("-> Updating best stats from epoch {0} := {1}".format(best_stats.epoch, best_val))
                if cfg.eval_higher_better:
                    self.best_acc = max(self.best_acc, best_val)
                else:
                    self.best_acc = min(self.best_acc, best_val)

        # loop through epochs
        scheduler.last_epoch = start_epoch

        return datasets, model, optimizer, scheduler, stats, loaded_path

    def run_training(self):
        datasets, model, optimizer, scheduler, stats, loaded_path = \
            self.initialize(self.cfg)
        
        start_epoch = scheduler.last_epoch
        for epoch in range(start_epoch,self.cfg.SOLVER.max_epochs):
            self.run_epoch(
                datasets,
                model, optimizer, scheduler, 
                epoch, stats, self.cfg)

    def trainvalidate(  self,
                        model,
                        stats,
                        epoch,
                        loader,
                        optimizer,
                        validation,
                        bp_var='objective',
                        metric_print_interval=5,
                        visualize_interval=100, 
                        visdom_env_root='trainvalidate',
                        **kwargs ):

        if validation:
            model.eval()
            trainmode = 'val'
        else:
            model.train()
            trainmode = 'train'

        t_start = time.time()

        # clear the visualisations on the first run in the epoch
        clear_visualisations = True

        # get the visdom env name
        visdom_env_imgs = visdom_env_root + "_images_" + trainmode

        n_batches = len(loader)
        for it, batch in enumerate(loader):
            last_iter = it==n_batches-1        

            # move to gpu where possible
            net_input = get_net_input(batch)

            # the forward pass
            if (not validation):
                preds = model(epoch, net_input)

                # make sure we dont overwrite something
                assert not any( k in preds for k in net_input.keys() )    
                preds.update(net_input) # merge everything into one big dict

                optimizer.zero_grad()
                gen_loss = preds['objective'].mean() # mean over the batch
                gen_loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    preds = model(epoch, net_input)

                # make sure we dont overwrite something
                assert not any( k in preds for k in net_input.keys() )    
                preds.update(net_input) # merge everything into one big dict

            # update the stats logger
            stats.update(preds,time_start=t_start,stat_set=trainmode)
            assert stats.it[trainmode]==it, "inconsistent stat iteration number!"

            # print textual status update
            if (it % metric_print_interval) == 0 or last_iter:
                stats.print(stat_set=trainmode,max_it=n_batches)
                
            # visualize results
            if (visualize_interval>0) and (it%visualize_interval)==0:
                with torch.no_grad():
                    model.module.visualize( visdom_env_imgs, trainmode, preds, stats, clear_env=clear_visualisations )
                # clear_visualisations = False

    def run_eval(self,cfg,model,stats,loader):
        eval_script, cache_vars, eval_vars = self.eval_zoo(
            cfg.DATASET.dataset_name)
        cached_preds = cache_preds(
            model, loader, stats=stats, cache_vars=cache_vars )
        results,_ = eval_script(
            cached_preds,eval_vars=eval_vars)
        stats.update(results,stat_set='test')
        stats.print(stat_set='test')