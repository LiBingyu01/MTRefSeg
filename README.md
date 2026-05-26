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

## Environment

For VLM-based methods, a common dependency reference is provided in:

```bash
pip install -r requirements_for_VLM.txt
```

Some sub-projects have their own dependencies, pretrained weights, dataset paths, and running scripts. Please follow the README file inside each corresponding subfolder.

## Notes

* The code packages in `VLMs/` and `LVLMs/` may have different environment requirements.
* Please unzip the corresponding package before running a specific baseline.
* Dataset paths, pretrained model paths, and output directories may need to be modified according to your local environment.
* For training, evaluation, checkpoint conversion, and model-specific details, please check the README in the corresponding subfolder.

## Acknowledgement

This repository builds upon multiple open-source referring segmentation, remote-sensing segmentation, and large vision-language model projects. We sincerely thank the original authors for their valuable contributions.
