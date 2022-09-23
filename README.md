# 3D Multibodies

## Installation

cd 3D-Multibodies
mkdir data

// Clone the SPIN data
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data

// Clone the SPIN pretrained checkpoint
mkdir -p data/pretrained
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data/pretrained

// Add files from <GOOGLE_LINK> to a directory data/3dmb
// Add files from <GOOGLE_LINK> to a directory data/crops
// Add files from <GOOGLE_LINK> to a directory data/pretrained

Download SMPL models as in config.py and place them in data/smpl

## Evaluation

CUDA_VISIBLE_DEVICES=0 python experiment.py \ 
    --visualize_interval 100 \
    --test_batch_size 75 \
    --num_workers 12 \
    --evaluate True \
    --MODEL.init_flow /scratch/code/3D-Multibodies/data/pretrained/101_00023_SOL.lr=1e-05_M.C.num_flows=20_M.C.hidden_dim=512/model_epoch_00000412.pth \
    --DATASET.dataset_name h36m_only \
    --DATASET.ambiguous False \
    --exp_dir /scratch/code/3D-Multibodies/data/pretrained/031_00022_DAT.ambiguous=False_M.l.loss_joint=25.0_M.l.loss_skel2d_modewise=1.0_M.l.loss_vertex=1.0

For H36M or AH36M set
--DATASET.dataset_name h36m_only

For 3DPW or A3DPW set
--DATASET.dataset_name 3dpw_only

For AH36M or A3DPW experiments set
--DATASET.ambiguous True \
--exp_dir /scratch/code/3D-Multibodies/data/pretrained/031_00010_DAT.ambiguous=True_M.l.loss_joint=25.0_M.l.loss_skel2d_modewise=1.0_M.l.loss_vertex=1.0





|             |   01_mpjpe |   01_reco |   05_mpjpe |   05_reco |   10_mpjpe |   10_reco |   25_mpjpe |   25_reco |
|:------------|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
| h36m_WEIGHT |     61.517 |   41.6263 |    60.4438 |   42.0243 |    59.7304 |   42.1845 |    58.5509 |   42.3683 |
| h36m        |     61.517 |   41.6263 |    59.7716 |   41.9884 |    59.1608 |   42.1022 |    58.236  |   42.1811 |

|              |   01_mpjpe |   01_reco |   05_mpjpe |   05_reco |   10_mpjpe |   10_reco |   25_mpjpe |   25_reco |
|:-------------|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
| ah36m_WEIGHT |    103.612 |   67.8314 |    95.3061 |   65.4571 |    92.4205 |   64.5733 |    88.7481 |   63.1861 |
| ah36m        |    103.612 |   67.8314 |    96.3664 |   67.0597 |    93.4299 |   65.9981 |    89.9254 |   64.2862 |

|             |   01_mpjpe |   01_reco |   05_mpjpe |   05_reco |   10_mpjpe |   10_reco |   25_mpjpe |   25_reco |
|:------------|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
| 3dpw_WEIGHT |    93.7943 |   59.8731 |    81.7215 |   56.714  |    78.7641 |   56.2927 |    75.2801 |   55.2845 |
| 3dpw        |    93.7943 |   59.8731 |    82.693  |   57.4905 |    79.936  |   57.0006 |    76.2492 |   55.9439 |

|              |   01_mpjpe |   01_reco |   05_mpjpe |   05_reco |   10_mpjpe |   10_reco |   25_mpjpe |   25_reco |
|:-------------|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
| a3dpw_WEIGHT |    149.671 |   78.3031 |    125.612 |   74.2592 |    116.774 |   73.5848 |    107.465 |   71.9949 |
| a3dpw        |    149.671 |   78.3031 |    132.169 |   77.6341 |    123.746 |   76.4744 |    112.871 |   74.0802 |