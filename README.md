# Anatomical Consistency and Adaptive Prior-informed Transformation for Multi-contrast MR Image Synthesis via Diffusion Model

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

This repo contains the code for our paper published in CVPR 2025: <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Shin_Anatomical_Consistency_and_Adaptive_Prior-informed_Transformation_for_Multi-contrast_MR_Image_CVPR_2025_paper.pdf"> **APT**  </a>.


![Image](https://github.com/user-attachments/assets/018e5b7d-8760-4047-b41c-2eb908691775)

> **Abstract:** Multi-contrast magnetic resonance (MR) images offer critical diagnostic information but are limited by long scan times and high cost. While diffusion models (DMs) excel in medical image synthesis, they often struggle to maintain anatomical consistency and utilize the diverse characteristics of multi-contrast MR images effectively. We propose APT, a unified diffusion model designed to generate accurate and anatomically consistent multi-contrast MR images. APT introduces a mutual information fusion module and an anatomical consistency loss to preserve critical anatomical structures across multiple contrast inputs. To enhance synthesis, APT incorporates a two-stage inference process: in the first stage, a prior codebook provides coarse anatomical structures by selecting appropriate guidance based on precomputed similarity mappings and Bezier curve transformations. The second stage applies iterative unrolling with weighted averaging to refine the initial output, enhancing fine anatomical details and ensuring structural consistency. This approach enables the preservation of both global structures and local details, resulting in realistic and diagnostically valuable synthesized images. Extensive experiments on public multi-contrast MR brain images demonstrate that our approach significantly outperforms state-of-the-art methods.

## For training MIF layer

- Enviroment Setting
```
$ cd ./adapter
$ conda create -n clip python=3.10 
$ conda activate clip
$ pip install -e . 
```
- For training
```
bash train.sh
```

## For training diffusion model
- Environment Setting

```
$ bash install.sh
$ conda activate APT
```
- For training
```
bash train_multi.sh
```

## For constructing prior codebook

```
python prior_codebook.py
```
> **Note**: For BRATS inference, the uploaded code already includes the pre-constructed prior codebook. Therefore, you do **not** need to run `prior_codebook.py` separately when using the BRATS dataset.

  ## For Inference
- 3 types of inference: One to three, two to two, three to one MR Image synthesis
```
bash run_1_to_3.sh
```
```
bash run_2_to_2.sh
```
```
bash run_3_to_1.sh
```


## Citation
If you found APT useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

@inproceedings{shin2025anatomical,
  title={Anatomical Consistency and Adaptive Prior-informed Transformation for Multi-contrast MR Image Synthesis via Diffusion Model},
  author={Shin, Yejee and Lee, Yeeun and Jang, Hanbyol and Son, Geonhui and Kim, Hyeongyu and Hwang, Dosik},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30918--30927},
  year={2025}
}

## Acknowledgments

* This repo is mainly based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [CLIP](https://github.com/openai/CLIP).
