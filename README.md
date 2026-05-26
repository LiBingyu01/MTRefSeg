# MTRefSeg Baseline Collection

This branch provides a collection of baseline implementations for **Multi-temporal Referring Segmentation (MTRS)** and related bi-temporal referring change segmentation experiments.

The repository is mainly organized into two parts:

```text
MTRefSeg/
├── VLMs/                     # Conventional vision-language referring segmentation baselines
├── LVLMs/                    # Large vision-language model based baselines
├── requirements_for_VLM.txt  # Common environment reference for VLM baselines
└── README.md                 # This file
```

## VLM-based Baselines

The `VLMs/` folder contains conventional vision-language referring segmentation models adapted to the bi-temporal setting, including CRIS, FIANet, LAVT, RMSIN, RSRefSeg, RefSegFormer, LGCE, and related methods.

Please refer to the detailed instructions in:

```text
VLMs/README.md
VLMs/README_CN.md
```

## LVLM-based Baselines

The `LVLMs/` folder contains large vision-language model based segmentation and grounding frameworks adapted to dual-time image inputs, including GeoPixel, LISA, GSVA, GroundingLMM, SegEarth-R1, UniChange, UniGeoSeg, and related methods.

Please refer to the detailed instructions in:

```text
LVLMs/README.md
LVLMs/README_CN.md
```

## Notes

* The code packages in `VLMs/` and `LVLMs/` have different environment requirements.
* Please unzip the corresponding package before running a specific baseline.
* Dataset paths, pretrained model paths, and output directories may need to be modified according to your local environment.
* For training, evaluation, checkpoint conversion, and model-specific details, please check the README in the corresponding subfolder.
* Note that the current **GeoPixel-based** baseline does not achieve satisfactory performance in our experiments. We welcome suggestions, discussions, and improved reproduction settings. Please contact us at libingyu0205@mail.ustc.edu.cn or libingyu0205@163.com.

## Acknowledgement

This repository builds upon multiple open-source referring segmentation, remote-sensing segmentation, and large vision-language model projects. We sincerely thank the original authors for their valuable contributions.

We especially acknowledge the following open-source projects:

* [CRIS](https://github.com/DerrickWang005/CRIS.pytorch)
* [LAVT](https://github.com/yz93/LAVT-RIS)
* [FIANet](https://github.com/Shaosifan/FIANet)
* [RMSIN](https://github.com/Lsan2401/RMSIN)
* [RSRefSeg](https://github.com/KyanChen/RSRefSeg)
* [RefSegFormer / Robust Referring Segmentation](https://github.com/jianzongwu/robust-ref-seg)
* [RRSIS / LGCE](https://gitlab.lrz.de/ai4eo/reasoning/rrsis)
* [LISA](https://github.com/dvlab-research/LISA)
* [GSVA](https://github.com/LeapLabTHU/GSVA)
* [GroundingLMM / GLaMM](https://github.com/mbzuai-oryx/groundingLMM)
* [GeoPixel](https://github.com/mbzuai-oryx/GeoPixel)
* [SegEarth-R1](https://github.com/earth-insights/SegEarth-R1)
* [UniChange](https://github.com/Erxucomeon/UniChange)
* [UniGeoSeg](https://github.com/MiliLab/UniGeoSeg)

