# Adaptive sparse attention module based on reciprocal nearest neighbors
Zhonggui Sun, Can Zhang, Mingzhu Zhang.

This code is based on the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) from [OpenMMlab](https://openmmlab.com/) 
__________
**Contents**
- [Abstract](#abstract)
- [Brief Introduction](#brief-introduction)
- [Usage](#usage)
- [Results](#results)
  - [Quantitative Results](#quantitative-results)
  - [Qualitative Results](#qualitative-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Abstract
The attention mechanism has become a crucial technique in deep feature representation for computer vision tasks. Using a similarity matrix, it enhances the current feature point with global context from the feature map of the network. However, the indiscriminate utilization of all information can easily introduce some irrelevant contents, inevitably hampering performance. In response to this challenge, sparsing, a common information filtering strategy, has been applied in many related studies. Regrettably, their filtering processes often lack reliability and adaptability. To address this issue, we first define an adaptive-reciprocal nearest neighbors (A-RNN) relationship. In identifying neighbors, it gains flexibility through learning adaptive thresholds. Additionally, by introducing a reciprocity mechanism, the reliability of neighbors is ensured. Then, we use A-RNN to rectify the similarity matrix in the conventional attention module. In the specific implementation, to distinctly consider non-local and local information, we introduce two blocks: the non-local sparse constraint block (NLSCB) and the local sparse constraint block (LSCB). The former utilizes A-RNN to sparsify non-local information, while the latter uses adaptive thresholds to sparsify local information. As a result, an adaptive sparse attention (ASA) module is achieved, inheriting the advantages of flexibility and reliability from A-RNN. In the validation for the proposed ASA module, we use it to replace the attention module in NLNet and conduct experiments on semantic segmentation benchmarks including Cityscapes, ADE20K and PASCAL VOC 2012. With the same backbone (ResNet101), our ASA module outperforms the conventional attention module and its some state-of-the-art (SOTA) variants.

## Introduction
<div align=center><img src="https://github.com/CanZhang01/ASA/blob/main/Fig.1.png"/></div>

Our contributions are as follows:
1) We define an adaptive reciprocal nearest neighbors (A-RNN) relationship that is both reliable and flexible.
2) We propose an adaptive sparse attention (ASA) module. Its similarity matrix is obtained via two designed blocks: NLSCB and LSCB. The former utilizes A-RNN to sparsify non-local information, while the latter sparsifies local information with adaptive thresholds.
3) Preliminary experiments on semantic segmentation validate the effectiveness of the proposed ASA module.

<div align=center><img src="https://github.com/CanZhang01/ASA/blob/main/Fig.2.png"/></div>

## Usage
Please refer to [MMsegmentation](https://mmsegmentation.readthedocs.io/en/latest/) help documentation.

## Result
### Quantitative Results
...to be continued after being accepted
### Qualitative Results
...to be continued after being accepted


## Acknowledgments
The authors would like to express their great thankfulness to the Associate Editor and the anonymous reviewers for their valuable comments and constructive suggestions. At the same time, they would like to express their sincere gratitude to the open-source semantic segmentation library MMSegmentation from openmmlab.
