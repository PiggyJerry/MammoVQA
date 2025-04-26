# MammoVQA: LVLMs for VQA on Mammograms

## Clone repository
```shell
git clone https://github.com/PiggyJerry/MammoVQA.git
```

## Prepare MammoVQA dataset
Download it from [figshare](), then unzip it and put under 'MammoVQA/'.

## Prepare compared LVLMs
Please follow the repositories of compared LVLMs ([BLIP-2\InstructBLIP](https://github.com/salesforce/LAVIS/tree/main),[LLaVA-Med](https://github.com/microsoft/LLaVA-Med),[LLaVA-NeXT-interleave](https://github.com/LLaVA-VL/LLaVA-NeXT),[Med-Flamingo](https://github.com/snap-stanford/med-flamingo),[MedDr](https://github.com/sunanhe/MedDr),[MedVInT_TD](https://github.com/xiaoman-zhang/PMC-VQA),[minigpt-4](https://github.com/Vision-CAIR/MiniGPT-4),[RadFM](https://github.com/chaoyi-wu/RadFM)) to prepare the weights and environments.
❗All the LLM weights should be put under 'MammoVQA/LLM/', except the weight of **MedVInT_TD** should be put under 'MammoVQA/Sota/MedVInT_TD/results/' and the weight of **RadFM** should be put under 'MammoVQA/Sota/RadFM-main/Quick_demo/'.
