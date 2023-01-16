# Model Zoo

|  Method  |  Pre-train Data & Checkpoint | Extra Label | Backbone | #Frame x Sample Rate | Script & Log & Checkpoint | mAP  |
| :------: | :--------------: | :---------: | :------: | :------------------: | :--: | ---- |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww) |   &cross;   |  ViT-S   |         16x4         |         [script](scripts/ava/videomae_vit_small_k400_pretrain/run.sh)/[log](https://drive.google.com/file/d/12OtNOd5kEEVlk9mpjojnO4xW1tXBVrZz/view?usp=share_link)/[checkpoint](https://drive.google.com/file/d/1gFJ8NnPPinpDO_caBg4CXR6fTDPOCVw_/view?usp=share_link)         | 22.5 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1ygjLRm1kvs9mwGsP3lLxUExhRo6TWnrx) |   &check;   |  ViT-S   |         16x4         |         [script](scripts/ava/videomae_vit_small_k400_pretrain+finetune/run.sh)/[log](https://drive.google.com/file/d/1sXITdzdWEPZ8qgwaaO02RM4JwPcKGJO5/view?usp=share_link)/[checkpoint](https://drive.google.com/file/d/1b9Lp64IF8vjhh19mjDdyoSSiZk6cFfKb/view?usp=share_link)         | 28.4 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1) |   &cross;   |  ViT-B   |         16x4         |         [script](scripts/ava/videomae_vit_base_k400_pretrain/run.sh)/[log](https://drive.google.com/file/d/1W0Vx-ItK7qD_b18vuqVY8Cld2QeZTqiV/view?usp=share_link)/[checkpoint](https://drive.google.com/file/d/19OG6aW-JKYtcgNiBLE4VXD6oAKAfklCB/view?usp=share_link)         | 26.7 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1MzwteHH-1yuMnFb8vRBQDvngV1Zl-d3z) |   &check;   |  ViT-B   |         16x4         |         [script](scripts/ava/videomae_vit_base_k400_pretrain+finetune/run.sh)/[log](https://drive.google.com/file/d/1LoYIEpMY3b__vPSyZw3AI-gBG_GeOSUo/view?usp=share_link)/[checkpoint](https://drive.google.com/file/d/1-YgaSSM9xWc4V_HjVG-FH0LMA6KV-Qdh/view?usp=share_link)         | 31.8 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3) |   &cross;   |  ViT-L   |         16x4         |         [script](scripts/ava/videomae_vit_large_k400_pretrain/run.sh)         | 34.3 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1jX1CiqxSkCfc94y8FRW1YGHy-GNvHCuD) |   &check;   |  ViT-L   |         16x4         |         [script](scripts/ava/videomae_vit_large_k400_pretrain+finetune/run.sh)         | 37.0 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/1AJQR1Rsi2N1pDn9tLyJ8DQrUREiBA1bO) |   &cross;   |  ViT-H   |         16x4         |         [script](scripts/ava/videomae_vit_huge_k400_pretrain/run.sh)         | 36.5 |
| VideoMAE | Kinetics-400 [checkpoint](https://drive.google.com/file/d/104ouJZxSVPSAm0LwJXd6IzjdA_RGLqZi) |   &check;   |  ViT-H   |         16x4         |         [script](scripts/ava/videomae_vit_huge_k400_pretrain+finetune/run.sh)         | 39.5 |


### Note:

- Extra Label &cross; means only **unlabelled** data is used during the pre-training phase.