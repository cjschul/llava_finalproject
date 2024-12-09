# LLaVA - Aerial Landscape Analysis Project

A reimplementation and extension of the Visual Instruction Tuning [1] methodology, focusing on aerial landscape analysis for emergency landing scenarios.

## Overview

This project extends LLaVA (Large Language and Vision Assistant) capabilities by applying it to aerial imagery analysis. The implementation combines CLIP for vision encoding and LLaMA (1B) for language decoding, with a specialized focus on terrain description and analysis.

## Technical Implementation

**Model Architecture**
- Vision Encoder: CLIP
- Language Decoder: LLaMA 1B
- Feature Alignment: Projection matrix for modality connection

**Training Pipeline**
- Pre-training: 175K image-caption pairs
- Fine-tuning: 150K multimodal instruction-following samples
- Evaluation: GPT-4 based benchmarking[2]

## Dataset

The project utilizes:
- COCO dataset for basic instruction tuning [2]
- SkyView (Kaggle) dataset for aerial imagery finetuning [3]

## Evaluation

Performance evaluation is conducted through:
- Comparative analysis with GPT-4
- Scoring based on accuracy and precision in landscape description

## Future Development

- Integration with larger models (Vicuna 7B)
- Expansion to satellite imagery
- Enhanced evaluation through human annotation
- Integration with OpenFlamingo for comparative analysis

## Requirements

- Python 3.x
- PyTorch
- Transformers
- CLIP
- Jupyter Notebook

## License

This project is available under standard open-source licensing terms.

Citations:
[1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. arXiv preprint arXiv:2304.08485, 2023.
[2] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common Objects in Context. arXiv preprint arXiv:1405.0312, 2014.
[3] Ankit Singh. SkyView: An Aerial Landscape Dataset. Kaggle Dataset, 2023. Available at: https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

