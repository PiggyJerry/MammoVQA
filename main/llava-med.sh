source /home/jiayi/anaconda3/etc/profile.d/conda.sh
conda activate llava-med
python /home/jiayi/MammoVQA/Sota/LLaVA-Med-main/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --temperature 0.0