# Spec 001 — Detecção e alinhamento de face

**Status:** implementada · **Módulo:** `src/facerec/detection.py` · **Testes:** `tests/test_detection.py`

## Contexto
Toda imagem (cadastro, reconhecimento, datasets) passa primeiro pelo detector. Só faces
detectadas com confiança, em tamanho e nitidez suficientes, seguem para a rede neural.

## Requisitos
1. Detectar faces com **MTCNN** (`facenet-pytorch`), obtendo caixa, probabilidade e 5 landmarks.
2. Considerar como face apenas detecções com probabilidade ≥ `detection.prob_threshold` (padrão 0,95).
3. Alinhar a face conforme o modelo:
   - FaceNet → `crop`: recorte da caixa redimensionado para 160×160 (pré-processamento do treino original);
   - ArcFace → `similarity`: transformação de similaridade dos 5 landmarks para o template 112×112 do insightface.
4. Checagem de qualidade (aplicação):
   - menor lado da caixa < `min_face_px` (padrão 60 px) → `LowQualityImage`;
   - variância do Laplaciano do recorte < `blur_threshold` → `LowQualityImage` (imagem borrada).
5. Respeitar a orientação EXIF; arquivos que não são imagem → `InvalidImage`.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | imagem sem face confiável | `detect_single` | `NoFaceDetected` |
| CA2 | imagem com 2+ faces confiáveis | `detect_single` | `MultipleFacesDetected` |
| CA3 | face com lado < 60 px | `detect_single` | `LowQualityImage` ("aproxime-se") |
| CA4 | face borrada | `detect_single` | `LowQualityImage` ("desfocada") |
| CA5 | uma face boa | `detect_single`, modo `crop` | recorte uint8 160×160×3 |
| CA6 | uma face boa | modo `similarity` | recorte 112×112 com landmarks no template |
| CA7 | bytes inválidos | `load_image` | `InvalidImage` |

## Fora de escopo
Anti-spoofing (detecção de foto de foto / tela); rastreamento em vídeo.

## Decisões
Em **datasets** (pré-processamento) não se rejeita múltiplas faces: usa-se a maior, que é a
pessoa rotulada. No LFW, se nenhuma face for detectada, usa-se recorte central (prática
comum do protocolo) — contagens em `experiments/results/preprocess_*.json`.
