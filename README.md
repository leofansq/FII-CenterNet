# FII-CenterNet
## How to use
### Training
* Training on Pascal VOC
  ```
  python main.py ctdet --exp_id voc_exp --batch_size 32 --gpus 0,1 --attention --wh_weight=0.1 --lr=1.25e-4 --num_epochs=140 --lr_step='90,120'
  ```
* Training on KITTI
  ```
  python main.py ctdet --exp_id kitti_exp --dataset=kitti --batch_size 8 --gpus 0 --attention --wh_weight=0.1 --lr=1e-4
  ```
  > For Evaluation: rename ./src/lib/datasets/dataset/kitti_half.py to kitti.py
  > For Testing: rename ./src/lib/datasets/dataset/kitti_full.py to kitti.py

### Evaluation
* Pascal VOC
  ```
  python test.py --exp_id voc_exp --not_prefetch_test ctdet --load_model /your_path/your_model.pth --attention --flip_test --trainval
  ```
  ```
  python tools/reval.py /your_path/results.json
  ```
* KITTI
  ```
  python test.py --exp_id kitti_exp --not_prefetch_test ctdet --dataset=kitti --load_model /your_path/your_model.pth --attention --flip_test
  ```
  ```
  ./tools/kitti_eval/evaluate_object_3d_offline ../data/kitti/training/Labels/ /your_path/results/
  ```

### Testing on KITTI
```
python test.py --exp_id kitti_full --not_prefetch_test ctdet --dataset=kitti --load_model /your_path/model_last.pth --attention --flip_test --trainval
```

### Demo
```
python demo.py ctdet --demo /.../FII-CenterNet/kitti_images/ --load_model /your_path/your_model.pth --attention --debug=2
```