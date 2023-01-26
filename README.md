# STCAE
Spatial-Temporal Convolutional Attention for Mapping Functional Brain Networks [preprint_arxiv](https://arxiv.org/abs/2211.02315)

## Requirement
- pytorch
- numpy
- nilearn
- nibabel
- tqdm
- tensorboardX

## Training
- STCAE in an individual
```python
from utils.activate import STAIndividual
sta = STAIndividual(mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
                    img_path="/home/public/ExperimentData/HCP900/HCP_RestingonMNI/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz",
                    device="cuda",
		    model_path=None,
		    time_step=40,
		    out_map=64,
                    lr=0.0001) 
sta.load_img()
sta.fit(epochs=1)
sta.eval()

img2d = sta.predict(0)

sta.plot_net(img2d)
```

- STCAE in MOTOR task 

`python train.py --device cuda --epochs 5 --time_step 284 --task MOTOR --out_map 64 --load_num -1`
- MutiHeadSTCAE in hcp rest
 
 `nohup python train.py --model mutiheadstcae --encoder hcp_rest_1200_head8 --n_heads 8 --device cuda --img_path /home/public/ExperimentData/HCP900/hcp_rest/  --epochs 3 --time_step 1200 --task None --out_map 64 --load_num 40 > out.log 2>&1 &`

- MutiHeadSTCAE with sampling (HCP-rest)

`nohup python train.py --model mutiheadstcae --encoder hcp_rest_1200_head16_sample176 --n_heads 16 --device cuda --img_path /home/public/ExperimentData/HCP900/hcp_rest/  --epochs 3 --time_step 176 --task None --out_map 64 --load_num 40 --sample 1 --sample_num 176 > out.log 2>&1 &`
- MutiHeadSTCAE with sampling (ADHD200)

`nohup python train.py --load_dataset adhd --model mutiheadstcae --encoder adhd_rest_head16_sample176 --n_heads 16 --device cuda --img_path /home/public/ExperimentData/ADHD200/adhd/adhd40.npy --epochs 3 --time_step 176 --out_map 64 --sample_num 176 > out.log 2>&1 &`


## Result
The results can be seen at [HCP-rest](./HCP_rest.ipynb) and [HCP-task (motor)](./hcp_motor_0-corr-window_size_40.ipynb)

## Citing STCAE
```
@inproceedings{stcae,
  title={Spatial-Temporal Convolutional Attention for Mapping Functional Brain Networks},
  author={Liu, Yiheng and Ge, Enjie and Qiang, Ning and Liu, Tianming and Ge, Bao},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI 2023)},
  year={2023},
  organization={IEEE}
}
```
