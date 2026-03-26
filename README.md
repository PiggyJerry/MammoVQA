# [A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram Visual Question Answering](https://www.nature.com/articles/s41467-025-66507-z)

[![DOI](https://zenodo.org/badge/930210527.svg)](https://doi.org/10.5281/zenodo.17384739)

## Clone repository
```shell
git clone https://github.com/PiggyJerry/MammoVQA.git
cd MammoVQA
conda create -n mammovqa python==3.9
conda activate mammovqa

python -m pip install -r requirements.txt
```

## Prepare MammoVQA dataset
### Sub-Dataset-Links:
Sub-Datasets downloading URL:
| Dataset Name | Dataset Link | Paper Link | Access |
|--------------|--------------|------------|--------|
| BMCD | [Link](https://zenodo.org/records/5036062) | [Digital subtraction of temporally sequential mammograms for improved detection and classification of microcalcifications](https://link.springer.com/content/pdf/10.1186/s41747-021-00238-w.pdf) | Open Access |
| CBIS-DDSM | [Link](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) | [A curated mammography data set for use in computer-aided detection and diagnosis research](https://www.nature.com/articles/sdata2017177.pdf) | Open Access |
| CDD-CESM | [Link](https://www.kaggle.com/datasets/krinalkasodiya/new-cdd-cesm-classification-data) | [Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research](https://www.nature.com/articles/s41597-022-01238-0.pdf) | Open Access |
| DMID | [Link](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883) | [Digital mammography dataset for breast cancer diagnosis research (dmid) with breast mass segmentation analysis](https://link.springer.com/article/10.1007/s13534-023-00339-y) | Open Access |
| INbreast | [Link](https://www.kaggle.com/datasets/tommyngx/inbreast2012) | [Inbreast: toward a full-field digital mammographic database](https://repositorio.inesctec.pt/server/api/core/bitstreams/6bc3ba6a-1220-413d-9ffe-a89acb92652b/content) | Open Access |
| MIAS | [Link](https://www.kaggle.com/datasets/kmader/mias-mammography) | The mammographic images analysis society digital mammogram database | Open Access |
| CSAW-M | [Link](https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271) | [Csaw-m: An ordinal classification dataset for benchmarking mammographic masking of cancer](https://arxiv.org/pdf/2112.01330) | Credentialed Access |
| KAU-BCMD | [Link](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset?select=Birad5) | [King abdulaziz university breast cancer mammogram dataset (kau-bcmd)](https://www.mdpi.com/2306-5729/6/11/111) | Open Access |
| VinDr-Mammo | [Link](https://www.physionet.org/content/vindr-mammo/1.0.0/) | [Vindr-mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography](https://www.nature.com/articles/s41597-023-02100-7.pdf) | Credentialed Access |
| RSNA | [Link](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) | [RSNA: Radiological Society of North America. Rsna screening mammography breast cancer detection ai challenge](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge) | Open Access |
| EMBED | [Link](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | [The emory breast imaging dataset (embed): A racially diverse, granular dataset of 3.4 million screening and diagnostic mammographic images](https://pubs.rsna.org/doi/pdf/10.1148/ryai.220047) | Credentialed Access |
| DBT-Test | [Link](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/) | [ Detection of masses and architectural distortions in digital breast tomosynthesis: a publicly available dataset of 5,060 patients and a deep learning model](https://arxiv.org/pdf/2011.07995) | Open Access |
| LAMIS | [Link](https://github.com/LAMISDMDB/LAMISDMDB_Sample) | [Lamis-dmdb: A new full field digital mammography database for breast cancer ai-cad researches](https://www.sciencedirect.com/science/article/abs/pii/S1746809423012569) | Credentialed Access |
| MM | [Link](https://data.mendeley.com/datasets/fvjhtskg93/1) | [Mammogram mastery: a robust dataset for breast cancer detection and medical education](https://www.sciencedirect.com/science/article/pii/S2352340924006000) | Open Access |
| NLBS | [Link](https://www.frdr-dfdr.ca/repo/dataset/cb5ddb98-ccdf-455c-886c-c9750a8c34c2) | [Full field digital mammography dataset from a population screening program](https://www.nature.com/articles/s41597-025-05866-0.pdf) | Open Access |

The json file of MammoVQA can be found in [Google Drive](https://drive.google.com/file/d/1R1xK6NpdOVVb5lUXIDkTFX1ysS3XBw6e/view?usp=sharing), after downloading it, unzip the file and put under `/Benchmark/`.

## Processing Dataset Codes and Files Linking:

| Dataset Name | Process Dataset Code |
|--------------|----------------------|
| BMCD | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/BMCD.ipynb |
| CBIS-DDSM | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/CBIS-DDSM.ipynb |
| CDD-CESM | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/CDD-CESM.ipynb |
| DMID | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/DMID.ipynb |
| INbreast | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/INbreast.ipynb |
| MIAS | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/MIAS.ipynb |
| CSAW-M | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/CSAW-M.ipynb |
| KAU-BCMD | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/KAU-BCMD.ipynb |
| VinDr-Mammo | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/VinDr-Mammo.ipynb |
| RSNA | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/rsna.ipynb |
| EMBED | https://github.com/PiggyJerry/MammoVQA/blob/main/preprocess/EMBED.ipynb |

After downloaded sub-datasets above, you have to use the correspoding processing code for it. Remember to change the dataset link in the code!!!

## Prepare compared LVLMs
If you only want to evaluate your model on MammoVQA, you can skip it.

Please follow the repositories of compared LVLMs ([BLIP-2\InstructBLIP](https://github.com/salesforce/LAVIS/tree/main),[LLaVA-Med](https://github.com/microsoft/LLaVA-Med),[LLaVA-NeXT-interleave](https://github.com/LLaVA-VL/LLaVA-NeXT),[Med-Flamingo](https://github.com/snap-stanford/med-flamingo),[MedDr](https://github.com/sunanhe/MedDr),[MedVInT_TD](https://github.com/xiaoman-zhang/PMC-VQA),[minigpt-4](https://github.com/Vision-CAIR/MiniGPT-4),[RadFM](https://github.com/chaoyi-wu/RadFM,[InternVL3](https://github.com/OpenGVLab/InternVL),[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),[MedGemma](https://github.com/Google-Health/medgemma)) to prepare the weights and environments.

❗All the LLM weights should be put under `MammoVQA/LLM/`, except the weight of **MedVInT_TD** should be put under `MammoVQA/Sota/MedVInT_TD/results/` and the weight of **RadFM** should be put under `MammoVQA/Sota/RadFM-main/Quick_demo/`.

## Finetune DiNOv2
You can finetune DiNOv2 on our MammoVQA training dataset and evaluate it on our test dataset. 

- S1. Download the DiNOv2 pre-trained weight here[https://github.com/facebookresearch/dinov2], and modify the DiNOv2 weight's path in the 67 line of `/MammoVQA/finetune/DiNOv2/models/image_encoder.py`.
- S2. `python /MammoVQA/finetune/DiNOv2/main.py` to train the model.
- S3. `python /MammoVQA/finetune/DiNOv2/eval.py` to evaluate the model, and you can get the result file in `/MammoVQA/Result/DiNOv2.json`.
- S4. `python /MammoVQA/Eval/Visiononly.py` to calculate metrics.


## Quick Start:

For quick start, you can check the `Quick_demo` path.
We demonstrate a simple diagnosis case here to show how to inference on MammoVQA with our LLaVA-Mammo.   
Feel free to modify it as you want.

- S1. Download [Model checkpoint](https://drive.google.com/file/d/1uFCrOTbsvug8YZoHKR7wlvoTSwzB32EY/view?usp=sharing) of LLaVA-Mammo, and unzip it to `Quick_demo` path.
- S2. `python /MammoVQA/Quick_demo/eval.py` to inference, and you can get the result file in `/MammoVQA/Result/LLaVA-Mammo.json`.
- S3. `python /MammoVQA/Eval/LLM.py` to calculate metrics.
## Citation
```
@article{zhu2025benchmark,
  title={A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram Visual Question Answering},
  author={Zhu, Jiayi and Huang, Fuxiang and Luo, Qiong and Chen, Hao},
  journal={Nature Communications},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
