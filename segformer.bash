conda activate segformer
segformer_path=./SegFormer

python datasets/tools/extract_masks.py \
    --data_root data/nuplan/processed/mini \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/nuplan_example_scenes.txt \
    --process_dynamic_mask