# imports here ...
import os
import numpy as np
import torch
import torch.nn.functional as F

from tabulate import tabulate
from accuracies import AccuracyMetrics
import matplotlib.pyplot as plt
import config

def eval_zoo(dataset):
	dataset_name = config.DATASET_LIST[dataset.dataset_key]
	eval_script = lambda *args, **kargs: run_evaluation(
		dataset_name, *args, **kargs)

	cache_vars  = [
		'M01_mpjpe',
		'M01_r_error',
		'M05_mpjpe',
		'M05_r_error',
		'WM05_mpjpe',
		'WM05_r_error',
		'M10_mpjpe',
		'M10_r_error',
		'WM10_mpjpe',
		'WM10_r_error',
		'M25_mpjpe',
		'M25_r_error',
		'WM25_mpjpe',
		'WM25_r_error',
		'M100_mpjpe',
		'M100_r_error',  
	]

	eval_vars   = [
		f'EVAL_{dataset_name}_M01_mpjpe', 
		f'EVAL_{dataset_name}_M01_r_error', 
		f'EVAL_{dataset_name}_M05_mpjpe', 
		f'EVAL_{dataset_name}_M05_r_error', 
		f'EVAL_{dataset_name}_WM05_mpjpe', 
		f'EVAL_{dataset_name}_WM05_r_error', 
		f'EVAL_{dataset_name}_M10_mpjpe', 
		f'EVAL_{dataset_name}_M10_r_error', 
		f'EVAL_{dataset_name}_WM10_mpjpe', 
		f'EVAL_{dataset_name}_WM10_r_error', 
		f'EVAL_{dataset_name}_M25_mpjpe', 
		f'EVAL_{dataset_name}_M25_r_error', 
		f'EVAL_{dataset_name}_WM25_mpjpe', 
		f'EVAL_{dataset_name}_WM25_r_error', 
		f'EVAL_{dataset_name}_M100_mpjpe', 
		f'EVAL_{dataset_name}_M100_r_error', 
	]
	
	return eval_script, cache_vars, eval_vars

def run_evaluation(dataset_name, cache_vars, eval_vars=None):

	"""Run evaluation on the datasets and metrics we report in the paper. """

	eval_result = {}
	for cache_name, cache_val in cache_vars.items():
		eval_result[f"EVAL_{dataset_name}_{cache_name}"] = cache_val.mean()
	
	return eval_result, None