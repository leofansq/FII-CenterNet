* Only modified DLA-34 is used in FII-CenterNet. The realated code is in [pose_dla_dcn.py](./pose_dla_dcn.py).
* Files named 'exp_xxx.py' is the code used in ablation experiments on KITTI validation set.
  * [exp_origin.py](./exp_origin.py) is the original code in CenterNet.
  * Self-branch is added to [exp_att.py](./exp_att.py)
  * Both self-branch and up-branch is added to [exp_db.py](./exp_db.py). Two normal fusion methods (sum & cat) are used.