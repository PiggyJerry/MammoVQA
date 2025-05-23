# MammoVQA: A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram Visual Question Answering

## Clone repository
```shell
git clone https://github.com/PiggyJerry/MammoVQA.git

conda create -n mammovqa python==3.9
conda activate mammovqa

python -m pip install -r requirements.txt
```

## Prepare MammoVQA dataset
### Sub-Dataset-Links:
Sub-Datasets downloading URL:
    
| Dataset Name | Link | Access |
|-----|---------------|--------|
| BMCD | https://zenodo.org/records/5036062 | Open Access |
| CBIS-DDSM | https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset | Open Access |
| CDD-CESM | https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8 | Open Access |
| DMID | https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883 | Open Access |
| INbreast | [https://www.physionet.org/content/vindr-spinexr/1.0.0/](https://www.kaggle.com/datasets/tommyngx/inbreast2012) | Open Access |
| MIAS | [https://physionet.org/content/vindr-pcxr/1.0.0/](https://www.kaggle.com/datasets/kmader/mias-mammography) | Open Access |
| CSAW-M | https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271 | Credentialed Access |
| KAU-BCMD | https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset?select=Birad5 | Open Access |
| VinDr-Mammo | https://www.physionet.org/content/vindr-mammo/1.0.0/ | Credentialed Access |
| RSNA | https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data | Open Access |
| EMBED | [https://physionet.org/content/mimic-cxr/2.0.0](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | Credentialed Access |

The json file of MammoVQA can be found in [Google Drive](https://drive.google.com/file/d/1eXgk5aJy8eHqJQpxbQVSOxnAUXKQRrda/view?usp=drive_link), after downloading it, unzip the file and put into `/Benchmark`.

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

Please follow the repositories of compared LVLMs ([BLIP-2\InstructBLIP](https://github.com/salesforce/LAVIS/tree/main),[LLaVA-Med](https://github.com/microsoft/LLaVA-Med),[LLaVA-NeXT-interleave](https://github.com/LLaVA-VL/LLaVA-NeXT),[Med-Flamingo](https://github.com/snap-stanford/med-flamingo),[MedDr](https://github.com/sunanhe/MedDr),[MedVInT_TD](https://github.com/xiaoman-zhang/PMC-VQA),[minigpt-4](https://github.com/Vision-CAIR/MiniGPT-4),[RadFM](https://github.com/chaoyi-wu/RadFM)) to prepare the weights and environments.

❗All the LLM weights should be put under `MammoVQA/LLM/`, except the weight of **MedVInT_TD** should be put under `MammoVQA/Sota/MedVInT_TD/results/` and the weight of **RadFM** should be put under `MammoVQA/Sota/RadFM-main/Quick_demo/`.

## Quick Start:

For quick start, you can check the `Quick_demo` path.
We demonstrate a simple diagnosis case here to show how to inference on MammoVQA with our LLaVA-Mammo.   
Feel free to modify it as you want.

- S1. Download [Model checkpoint](https://drive.google.com/file/d/1uFCrOTbsvug8YZoHKR7wlvoTSwzB32EY/view?usp=sharing) of LLaVA-Mammo, and unzip it to `Quick_demo` path.
- S2. `python /MammoVQA/Quick_demo/main.py` to inference, you can get the result file in `/MammoVQA/Result/LLaVA-Mammo.json`.
- S3. `python /MammoVQA/Eval/Output_score_combine.py` to calculate metrics.
