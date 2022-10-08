# STCAE
Spatial-Temporal Convolutional Attention Encoder

## Requirement
- pytorch
- numpy
- nilearn
- nibabel
- tqdm
- tensorboardX

## Training
- STCAE in MOTOR task `python train.py --device cuda --epochs 5 --time_step 284 --task MOTOR --out_map 64 --load_num -1`
- MutiHeadSTCAE in hcp rest `nohup python train.py --model mutiheadstcae --encoder hcp_rest_1200_head8 --n_heads 8 --device cuda --img_path /home/public/ExperimentData/HCP900/hcp_rest/  --epochs 3 --time_step 1200 --task None --out_map 64 --load_num 40 > out.log 2>&1 &`
