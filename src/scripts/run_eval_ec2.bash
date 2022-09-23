#!/bin/bash

build_cmdstr() {
    dataset=$1
    gpu=$2
    echo "You are evaluating $dataset on GPU $gpu"

    cmd_list="--visualize_interval 100 \
    --batch_size 75 \
    --test_batch_size 75 \
    --num_workers 12 \
    --gpu_idx [0] \
    --evaluate True \
    --MODEL.init_flow /scratch/code/3D-Multibodies/data/pretrained/101_00023_SOL.lr=1e-05_M.C.num_flows=20_M.C.hidden_dim=512/model_epoch_00000412.pth"


    if [[ "$dataset" == "h36m" || "$dataset" == "ah36m" ]]; then
        cmd_list="$cmd_list --DATASET.dataset_name h36m_only"
    elif [[ "$dataset" == "3dpw" || "$dataset" == "a3dpw" ]]; then
        cmd_list="$cmd_list --DATASET.dataset_name 3dpw_only"
    else
        echo "ERROR. Dataset name '$dataset' is not recognized"
        exit
    fi


    if [[ "$dataset" == "h36m" || "$dataset" == "3dpw" ]]; then
        cmd_list="$cmd_list --DATASET.ambiguous False \
    --exp_dir /scratch/code/3D-Multibodies/data/pretrained/031_00022_DAT.ambiguous=False_M.l.loss_joint=25.0_M.l.loss_skel2d_modewise=1.0_M.l.loss_vertex=1.0"
    elif [[ "$dataset" == "ah36m" || $dataset=="a3dpw" ]]; then
        cmd_list="$cmd_list --DATASET.ambiguous True \
    --exp_dir /scratch/code/3D-Multibodies/data/pretrained/031_00010_DAT.ambiguous=True_M.l.loss_joint=25.0_M.l.loss_skel2d_modewise=1.0_M.l.loss_vertex=1.0"
    else
        echo "ERROR. Dataset name '$dataset' is not recognized"
        exit
    fi
        
    func_result="CUDA_VISIBLE_DEVICES=$gpu python experiment.py $cmd_list"
}

# exec python experiment.py $cmd_list 

# tmux kill-session -t multibodies_exps
tmux new-session -d -s multibodies_exps;
tmux split-window -h \; split-window -v \; select-pane -L \; split-window -v\;

declare -a arr=("h36m" "ah36m" "3dpw" "a3dpw")
arraylength=${#arr[@]}

echo "$arraylength"

for (( i=0; i<${arraylength}; i++ ));
do
    dataset=${arr[i]}
    gpu=$i
    build_cmdstr "$dataset" "$gpu"
    echo $func_result
    # tmux set pane-border-status top
    # tmux select-pane -t "$i" -T "$dataset: GPU $gpu";    
    tmux send "$func_result" ENTER;
done

tmux a;