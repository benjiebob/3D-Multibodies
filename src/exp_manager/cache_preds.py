import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import time
import sys
import copy
import torch

from tqdm import tqdm

from .stats import Stats
from .utils import pprint_dict, has_method, get_net_input

def cache_preds(model,loader,cache_vars=None,stats=None):

    print("caching model predictions: %s" % str(cache_vars) )
    
    model.eval()
    
    trainmode = 'test'

    t_start = time.time()

    iterator = loader.__iter__()

    cached_preds = []

    cache_size = 0. # in GB ... counts only cached tensor sizes

    n_batches = len(loader)
    
    with tqdm(total=n_batches,file=sys.stdout) as pbar:
        for it, batch in enumerate(loader):

            last_iter = it==n_batches-1

            # move to gpu and cast to Var
            net_input = get_net_input(batch)
            
            with torch.no_grad():
                preds = model(999, net_input, cache_mode=True) # fake epoch
                # preds = model(**net_input)

            assert not any( k in preds for k in net_input.keys() ) 
            preds.update(net_input) # merge everything into one big dict        
            
            if stats is not None:
                stats.update(preds,time_start=t_start,stat_set=trainmode)
                assert stats.it[trainmode]==it, "inconsistent stat iteration number!"                            

            # restrict the variables to cache
            if cache_vars is not None:
                preds = {k:preds[k] for k in cache_vars if k in preds}

            # ... gather and log the size of the cache
            for k in preds:                
                if has_method(preds[k],'cuda'):
                    preds[k] = preds[k].data.cpu()
                    cache_size += preds[k].numpy().nbytes / 1e9

            cached_preds.append(preds)

            concat_cache = concatenate_cache(cached_preds)
            p_dict = {}
            for print_key in ["M01_mpjpe", "WM05_mpjpe", "WM10_mpjpe", "WM25_mpjpe"]:
                p_dict[print_key] = concat_cache[print_key].mean()
            pbar.set_postfix(p_dict)


            # pbar.set_postfix(cache_size="%1.2f GB"%cache_size)
            # pbar.set_postfix({cache_size="%1.2f GB"%cache_size)
            pbar.update(1)

    # concatenate cache along 0 dim
    cached_preds_cat = concatenate_cache(cached_preds)

    return cached_preds_cat

def concatenate_cache(cached_preds):
    flds = list(cached_preds[0].keys())
    cached_preds_concat = {}
    for fld in flds:
        classic_cat = True
        if type(cached_preds[0][fld][0])==str:
            classic_cat = True
        else:
            try:
                cached_preds_concat[fld] = torch.cat([c[fld] for c in cached_preds], dim=0)
                classic_cat = False
            except:
                pass
        if classic_cat:
            cached_preds_concat[fld] = \
                    [x for c in cached_preds for x in c[fld]]
    
    return cached_preds_concat


    