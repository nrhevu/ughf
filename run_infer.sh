python infer.py \
    --afine-path  /home/vu.nguyen/workspace/nvidia/IQA-metrics/AFINE/afine.pth \
    --clip-path /home/vu.nguyen/workspace/nvidia/IQA-metrics/AFINE/ViT-B-32.pt \
    --dis-img-path /home/vu.nguyen/workspace/nvidia/IQA-metrics/SADHI-metrics/data/DiffIQA/Validation/images/03/000011x398y812_03.png \
    --ref-img-path /home/vu.nguyen/workspace/nvidia/IQA-metrics/SADHI-metrics/data/DiffIQA/Validation/images/Original/000011x398y812.png

python tests/test_clip.py \
    --afine-path  /home/vu.nguyen/workspace/nvidia/IQA-metrics/AFINE/afine.pth \
    --clip-path /home/vu.nguyen/workspace/nvidia/IQA-metrics/AFINE/ViT-B-32.pt \
    --dis-img-path /home/vu.nguyen/workspace/nvidia/IQA-metrics/SADHI-metrics/data/DiffIQA/Validation/images/03/000011x398y812_03.png \
    --ref-img-path /home/vu.nguyen/workspace/nvidia/IQA-metrics/SADHI-metrics/data/DiffIQA/Validation/images/Original/000011x398y812.png