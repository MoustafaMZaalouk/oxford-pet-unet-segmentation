| Model | pet IoU | background IoU | border IoU | mIoU | pixel acc |
|---|---|---|---|---|---|
| U-Net from scratch (7.85M params) | 77.4 [76.8, 78.0] | 87.3 [86.9, 87.7] | 42.6 [42.2, 43.0] | 69.1 [68.7, 69.5] | 87.8 [87.6, 88.1] |
| Frozen ResNet-18 (ImageNet) + linear probe (771 params) | 74.0 [73.5, 74.4] | 83.7 [83.4, 84.0] | 21.2 [20.9, 21.4] | 59.6 [59.3, 59.9] | 83.9 [83.7, 84.1] |
