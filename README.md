# 3D Multibodies

Code for the paper 3D Multibodies, NeurIPS 2020 Spotlight.

- [Paper](https://arxiv.org/abs/2011.00980)
- [Project Page](https://sites.google.com/view/3dmb/home)

## Installation

```
cd 3D-Multibodies
mkdir data

// Clone the SPIN data
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data

// Clone the SPIN pretrained checkpoint
mkdir -p data/pretrained
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data/pretrained
```
Download the 3DMB data from [Google Drive](https://drive.google.com/drive/folders/1O10yovP5Q4Dj1EugnTnPdJ59V_a6en98?usp=sharing)

- Add files from `3dmb` to a directory `data/3dmb`
- Add files from `crops` to a directory `data/crops`
- Add files from `pretrained` to the `data/pretrained` used above

Download SMPL models as in `src/config.py` and place them in `data/smpl`


## Evaluation
```
cd src
CUDA_VISIBLE_DEVICES=0 python experiment.py \ 
    --visualize_interval 100 \
    --test_batch_size 75 \
    --num_workers 12 \
    --evaluate True \
    --MODEL.init_flow ../data/pretrained/normflow/model_epoch_00000412.pth \
    --DATASET.dataset_name h36m_only \
    --DATASET.ambiguous False \
    --exp_dir ../data/pretrained/standard
```
For H36M or AH36M set
```
--DATASET.dataset_name h36m_only
```

For 3DPW or A3DPW set
```
--DATASET.dataset_name 3dpw_only
```

For AH36M or A3DPW experiments set
```
--DATASET.ambiguous True \
--exp_dir ../data/pretrained/ambiguous
```

## Run All

To run the complete set of evaluations, install `tmux` and use the script
```
cd src
bash scripts/run_eval_ec2.bash
```

## Expected Results


|             |   01_mpjpe |   01_reco |   05_mpjpe |   05_reco |   10_mpjpe |   10_reco |   25_mpjpe |   25_reco |
|:------------|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
| h36m        |     61.517 |   41.6263 |    59.7716 |   41.9884 |    59.1608 |   42.1022 |    58.236  |   42.1811 |
| 3dpw_WEIGHT |    93.7943 |   59.8731 |    81.7215 |   56.714  |    78.7641 |   56.2927 |    75.2801 |   55.2845 |
| ah36m_WEIGHT |    103.612 |   67.8314 |    95.3061 |   65.4571 |    92.4205 |   64.5733 |    88.7481 |   63.1861 |
| a3dpw_WEIGHT |    149.671 |   78.3031 |    125.612 |   74.2592 |    116.774 |   73.5848 |    107.465 |   71.9949 |

## Citation

```
} 
@inproceedings{biggs2020multibodies,
  author = "Biggs, Benjamin and Ehrhart, S{\'{e}}bastien and Joo, Hanbyul and Graham, Benjamin and Vedaldi, Andrea and Novotny, David",
  title = "{3D} Multibodies: Fitting Sets of Plausible {3D} Models to Ambiguous Image Data",
  booktitle = "NeurIPS",
  year = "2020",
}
```

## Acknowledgements
Much of the code here has been borrowed from the excellent [SPIN](https://github.com/nkolot/SPIN) repository.