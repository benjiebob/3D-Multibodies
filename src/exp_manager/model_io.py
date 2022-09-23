import pickle
import torch
import glob
import os

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "griddle.tools.stats":
            renamed_module = "exp_manager.stats"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def load_stats(flstats):
    try:
        stats, _ = pickle.load(open(flstats,'rb')) # dont load the config
    except:
        try:
            with open(flstats, "rb") as f:
                stats, _ = RenameUnpickler(f).load()
        except:
            print("Cant load stats! %s" % flstats)
            stats = None
    return stats

def get_model_path(fl):
    fl = os.path.splitext(fl)[0]
    flmodel = "%s.pth" % fl
    return flmodel

def get_optimizer_path(fl):
    fl = os.path.splitext(fl)[0]
    flopt = "%s_opt.pth" % fl
    return flopt

def get_stats_path(fl):
    fl = os.path.splitext(fl)[0]
    flstats = "%s_stats.pkl" % fl
    return flstats    

def save_model(model,stats,fl,optimizer=None,cfg=None):
    flstats = get_stats_path(fl)    
    flmodel = get_model_path(fl)
    print("saving model to %s" % flmodel)
    
    params = model.state_dict()
    for name, tensor in params.items():
        if hasattr(tensor, 'to_dense'):
            try:
                params[name] = tensor.to_dense()
            except RuntimeError:
                pass

    torch.save(params,flmodel)
    if optimizer is not None:
        flopt   = get_optimizer_path(fl)
        print("saving optimizer to %s" % flopt)
        torch.save(optimizer.state_dict(),flopt)
    print("saving model stats and cfg to %s" % flstats)
    pickle.dump((stats,cfg),open(flstats,'wb'))

def load_model(fl):
    flstats = get_stats_path(fl)    
    flmodel = get_model_path(fl)
    flopt   = get_optimizer_path(fl)
    model_state_dict = torch.load(flmodel)
    stats = load_stats(flstats)
    if os.path.isfile(flopt):
        optimizer = torch.load(flopt)
    else:
        optimizer = None

    return model_state_dict, stats, optimizer

def get_checkpoint(exp_dir,epoch):
    fl = os.path.join( exp_dir, 'model_epoch_%08d.pth' % epoch )
    return fl

def find_last_checkpoint(exp_dir,any_path=False):
    
    if any_path:
        exts = ['.pth','_stats.pkl','_opt.pth']
    else:
        exts = ['.pth']
    
    for ext in exts:
        fls = sorted( glob.glob( os.path.join( \
            exp_dir, 'model_epoch_'+'[0-9]'*8+ext )))
        if len(fls) > 0: break
    if len(fls)==0: 
        fl = None
    else: 
        fl = fls[-1][0:-len(ext)] + '.pth'

    return fl


def purge_epoch(exp_dir,epoch):
    model_path = get_checkpoint(exp_dir,epoch)
    to_kill = [ model_path,
                get_optimizer_path(model_path),
                get_stats_path(model_path) ]

    for k in to_kill:
        if os.path.isfile(k):
            print('deleting %s' % k)
            os.remove(k)