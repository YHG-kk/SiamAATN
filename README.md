# SiamAATN
## Enhanced UAV Tracking with Siamese Attention and Adaptive Anchor Proposal

Object tracking, a vital branch of computer vision, models an object's appearance and motion to predict its state and position. Traditional methods, often reliant on correlation filters, face challenges in complex scenarios. Recent advances in deep learning, particularly twin networks, have shown promise. This paper introduces a novel twin network-based tracker incorporating attention mechanisms and an adaptive anchor proposal strategy. By integrating multi-level features and applying scale-channel attention, our method achieves precise and rapid anchor suggestion. Evaluations on the UAV123 dataset series demonstrate enhanced tracking accuracy and robustness, particularly under rapid motion and scale changes.


![总框图](https://github.com/YHG-kk/SiamAATN/blob/main/img/%E6%80%BB%E6%A1%86%E5%9B%BE.svg)

##### Test in UAV123 series datasheet image sequence.

### <img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/truck.gif" alt="truck" style="zoom:80%;" /><img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/bike1.gif" alt="bike1" style="zoom:80%;" /><img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/uav1.gif" alt="uav1" style="zoom:80%;" />

# <img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/Pre_uav123_10fps.svg" alt="Pre_uav123_10fps" style="zoom:50%;" /><img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/Pre_uav123_20L.svg" alt="Pre_uav123_20L" style="zoom:50%;" /><img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/Suc_uav123_20L.svg" alt="Suc_uav123_20L" style="zoom:50%;" /><img src="https://github.com/YHG-kk/SiamAATN/blob/main/img/Suc_uav123_10fps.svg" alt="Suc_uav123_10fps" style="zoom:50%;" />

## Environment

Base:

Win11 25H2

cuda12.6

py3.6_cuda11.3_cudnn8_0

```
pip install -r requirements.txt
```

## Train

Datasheet:

[Got-10k]: http://got-10k.aitestunion.com/downloads

Start train:

```
python train_aatn.py # make sure datasheet and other file contents correctly
```

### Test and eval

```
python test.py --trackername SiamAATN --dataset YOUR-TEST-DATASHEET-NAME --snapshot YOUR-MODEL-PATH
python eval.py --dataset YOUR-TEST-DATASHEET --dataset_dir "YOUR-TEST-DATASHEET-PATH" --tracker_result_dir ./results --trackers YOUR-MODEL-PATH --vis #vis TO OPEN VISUALIZTION IF NEED
```

### Run a demo

```
python demo_aatn.py --snapshot YOUR-MODE-PATH --video_name YOUR-IMAGE-SEQUENCE 
python demo_aatn_web.py --snapshot YOUR-MODE-PATH # use web cam as input image
```

### Citation

```
@article{karakostas2025enhancing,
  title={Enhanced UAV Tracking with Siamese Attention and Adaptive Anchor Proposal},
  author={Yuan Luo,Hongguang Yan},
  journal={The Visual Computer},
  pages={1--10},
  year={2025},
  publisher={Springer}
}
```

### Acknowledgments

This code is based on 

[PySOT]: https://github.com/STVIR/pysot

. We sincerely thank all the contributors.
