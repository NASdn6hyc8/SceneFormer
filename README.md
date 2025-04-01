# SceneFormer

**Neural architecture search of Transformers for remote sensing scene classification**

## Usage

### Data Preparation

You need to download the UCM/NWPU45/AID and move images to labeled subfolders.

Here is a example of directory structure:

```
/PATH/TO/DATASET/
    class_1/
      img_1.jpeg
      img_2.jpeg
      ...
    class_2/
      img_3.jpeg
      ...
    ...
```
### SUPERNET TRAIN
You can train/finetune a supernet with following command:
```bulidoutcfg
python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /PATH/TO/DATASET/ --gp --change_qkv --mode super --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --output /OUTPUT_PATH --batch-size 128 --resume /PATH/TO/CHECKPOINT --data-set DATASET_NAME ''
```

## Acknowledgements

The codes are inspired by [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [HAT](https://github.com/mit-han-lab/hardware-aware-transformers), [DeiT](https://github.com/facebookresearch/deit), [SPOS](https://github.com/megvii-model/SinglePathOneShot).
