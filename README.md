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
### Supernet Train
You can train/fine-tune a supernet with following command:
```bulidoutcfg
python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /PATH/TO/DATASET/ --gp --change_qkv --mode super --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --output /OUTPUT_PATH --batch-size 128 --resume /PATH/TO/CHECKPOINT --data-set DATASET_NAME ''
```
### Evolutionary Search
You can search a Transformer architecture with following command:
```bulidoutcfg
python -m torch.distributed.launch --nproc_per_node=4 --use_env evolution.py --data-path /PATH/TO/DATASET/ --gp --change_qkv --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume /SUPERNET_CHECKPOINT --min-param-limits 6.0 --param-limits 7.0 --data-set DATASET_NAME --batch-size 128 --output_dir /OUTPUT_PATH --max-epochs 20 ''
```
### Test
Before testing the search results, you may need to comment out lines 325-327 in supernet_train.py. These lines remove the classification head from the model when fine-tuning the supernet, which should not be removed for testing. Additionally, you should create a yaml file similar to ./experiments/subnet/UCM_fold0.yaml based on your search results.

You can then test the search result with following command:
```bulidoutcfg
python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /PATH/TO/DATASET/ --gp --change_qkv --mode retrain --relative_position --dist-eval --cfg /SEARCH_RESULT --batch-size 128 --resume /PATH/TO/CHECKPOINT --data-set DATASET_NAME --eval ''
```

## Acknowledgements

The codes are inspired by [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [HAT](https://github.com/mit-han-lab/hardware-aware-transformers), [DeiT](https://github.com/facebookresearch/deit), [SPOS](https://github.com/megvii-model/SinglePathOneShot).
