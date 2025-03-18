# Anatomical Consistency and Adaptive Prior-informed Transformation for Multi-contrast MR Image Synthesis via Diffusion Model

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

This repo will contain the code for our paper published in CVPR 2025.


![Image](https://github.com/user-attachments/assets/018e5b7d-8760-4047-b41c-2eb908691775)

> **Abstract:** Multi-contrast magnetic resonance (MR) images offer critical diagnostic information but are limited by long scan times and high cost. While diffusion models (DMs) excel in medical image synthesis, they often struggle to maintain anatomical consistency and utilize the diverse characteristics of multi-contrast MR images effectively. We propose APT, a unified diffusion model designed to generate accurate and anatomically consistent multi-contrast MR images. APT introduces a mutual information fusion module and an anatomical consistency loss to preserve critical anatomical structures across multiple contrast inputs. To enhance synthesis, APT incorporates a two-stage inference process: in the first stage, a prior codebook provides coarse anatomical structures by selecting appropriate guidance based on precomputed similarity mappings and Bezier curve transformations. The second stage applies iterative unrolling with weighted averaging to refine the initial output, enhancing fine anatomical details and ensuring structural consistency. This approach enables the preservation of both global structures and local details, resulting in realistic and diagnostically valuable synthesized images. Extensive experiments on public multi-contrast MR brain images demonstrate that our approach significantly outperforms state-of-the-art methods.


## Code
- Code will be uploaded.


## Acknowledgments

* This repo is mainly based on [guided-diffusion](https://github.com/openai/guided-diffusion).
