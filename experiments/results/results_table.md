| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | EER | Observação |
|---|---|---|---|---|---|---|---|
| E01 | FaceNet (VGGFace2) | pretrained | LFW (10-fold) | 99.10% ± 0.43% | 99.10% | 0.93% | AUC 0.9991 |
| E02 | ArcFace iResNet-50 | pretrained | LFW (10-fold) | 99.85% ± 0.20% | 99.85% | 0.30% | AUC 0.9995 |
| E03 | FaceNet (VGGFace2) | partial | LFW (10-fold) | 99.10% ± 0.43% | 99.10% | 0.93% | AUC 0.9991 |
| E03 (última época) | FaceNet (VGGFace2) | partial | LFW (10-fold) | 98.40% ± 0.54% | 98.39% | 1.77% | AUC 0.9982 |
| E08 | MobileFaceNet (ArcFace loss) | pretrained | LFW (10-fold) | 99.67% ± 0.20% | 99.67% | 0.32% | AUC 0.9993 |
