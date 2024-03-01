# STCAE
- [Spatial-Temporal Convolutional Attention for Mapping Functional Brain Networks](https://ieeexplore.ieee.org/abstract/document/10230749)
- [Spatial-temporal convolutional attention for discovering and characterizing functional brain networks in task fMRI](https://www.sciencedirect.com/science/article/pii/S1053811924000144)

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
@inproceedings{liu2023spatial,
  title={Spatial-Temporal Convolutional Attention for Mapping Functional Brain Networks},
  author={Liu, Yiheng and Ge, Enjie and Qiang, Ning and Liu, Tianming and Ge, Bao},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}

@article{liu2024spatial,
  title={Spatial-temporal convolutional attention for discovering and characterizing functional brain networks in task fMRI},
  author={Liu, Yiheng and Ge, Enjie and Kang, Zili and Qiang, Ning and Liu, Tianming and Ge, Bao},
  journal={NeuroImage},
  pages={120519},
  year={2024},
  publisher={Elsevier}
}
```

## Related Work
- [SCAAE](https://github.com/WhatAboutMyStar/SCAAE)
```
@article{liu2022discovering,
  title={Discovering Dynamic Functional Brain Networks via Spatial and Channel-wise Attention},
  author={Liu, Yiheng and Ge, Enjie and He, Mengshen and Liu, Zhengliang and Zhao, Shijie and Hu, Xintao and Zhu, Dajiang and Liu, Tianming and Ge, Bao},
  journal={arXiv preprint arXiv:2205.09576},
  year={2022}
}

@article{liu2024mapping,
  title={Mapping dynamic spatial patterns of brain function with spatial-wise attention},
  author={Liu, Yiheng and Ge, Enjie and He, Mengshen and Liu, Zhengliang and Zhao, Shijie and Hu, Xintao and Qiang, Ning and Zhu, Dajiang and Liu, Tianming and Ge, Bao},
  journal={Journal of Neural Engineering},
  year={2024}
}
```
