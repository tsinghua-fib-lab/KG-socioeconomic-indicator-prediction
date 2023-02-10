# KG for socioeconomic indicator prediction

This is the codebase for "Hierarchical Knowledge Graph Learning Enabled Socioeconomic Indicator Prediction in Location-Based Social Network"(WWW'23).  
NYC dataset is included.

# Usage
Please first download "mob-adj.npy" from [here](https://cloud.tsinghua.edu.cn/f/351fa77cc997486183c1/?dl=1) and put it into "./data/data_ny/" folder.

Train:

```
bash run.sh
```
Evaluate:

```
python evaluate.py
```

# Reference
```
@inproceddings{zhou2023hierarchical,
title 	  = {Hierarchical Knowledge Graph Learning Enabled Socioeconomic Indicator Prediction in Location-Based Social Network},
author	  = {Zhou, Zhilun and Liu, Yu and Ding, Jingtao and Jin, Depeng and Li, Yong},
booktitle = {The Web Conference},
year      = {2023}}
```
