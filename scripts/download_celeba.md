# CelebA — download manual

O CelebA (Large-scale CelebFaces Attributes Dataset, MMLAB/CUHK) **não pode ser redistribuído**:
os termos de uso permitem apenas pesquisa não comercial e proíbem republicar as imagens.
Por isso este repositório **não** contém o dataset nem um script de download automático.

## Passos

1. Acesse a página oficial: <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html> e leia os termos.
2. Pelo link oficial (Google Drive / Baidu) baixe:
   - `Img/img_align_celeba.zip` (imagens alinhadas e recortadas, 178×218)
   - `Anno/identity_CelebA.txt` (identidade de cada imagem — IDs numéricos, sem nomes)
3. Extraia para:

   ```
   data/public/celeba/img_align_celeba/000001.jpg ...
   data/public/celeba/identity_CelebA.txt
   ```

4. Gere as faces alinhadas para o modelo (subconjunto de identidades com ≥ 15 imagens):

   ```
   python -m facerec.data.preprocess --dataset celeba --model facenet --celeba-max-identities 1000
   ```

5. Verifique a sobreposição com o LFW (depois de preparar também o LFW):

   ```
   python scripts/check_identity_overlap.py --model facenet
   ```

`data/public/` está no `.gitignore` e é bloqueado pelo hook `scripts/check_private_files.py`.

Citação: Z. Liu, P. Luo, X. Wang, X. Tang. *Deep Learning Face Attributes in the Wild.* ICCV 2015.
