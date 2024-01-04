# Overview

This is an ablation study of the Bayesian classification head we proposed in our preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023).

We started from the codebase from [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), which provides re-implementations of several SOTA SGG frameworks and the evaluation metrics.
(We highly recommend reading their original README.md first to get a basic understanding of this codebase.)


# Testing Results

We provide the testing results of predicate classifications(PLS) task on Visual Genome before and after we replace the flat classification head with
our Bayesian head at the last linear layer of three existing works: NeuralMotifs, VTransE and VCTree.


| Methods           | R@20  | R@50  | R@100 | mR@20 | mR@50 | mR@100 |
|-------------------|-------|-------|-------|-------|-------|--------|
| NeuralMotifs      | 58.5  | 65.2  | 67.0  | 15.7  | 14.8  | 16.1   |
| NeuralMotifs w/ [a]| 53.8  | 68.3  | 74.6  | 15.9  | 24.3  | 29.9   |
| VTransE           | 59.1  | 65.6  | 67.3  | 12.8  | 16.3  | 17.6   |
| VTransE w/ [a]    | 53.8  | 68.1  | 74.5  | 18.1  | 26.2  | 31.5   |
| VCTree            | 59.0  | 65.4  | 67.2  | 13.1  | 16.7  | 18.2   |
| VCTree w/ [a]     | 54.5  | 69.1  | 75.4  | 16.7  | 26.3  | 32.2   |

[a] means hierarchical relationships in this table.



# Training & Evaluation

After installing the prerequisites mentioned in the original repo, to use our Bayesian classification head instead of the original flat classification head, 
you simply need to set corresponding MODEL.ROI_RELATION_HEAD.PREDICTOR(see below) in the training cmd, and disable the bias option.

NeuralMotif w/Bayesian head: `MODEL.ROI_RELATION_HEAD.PREDICTOR MotifHierarchicalPredictor`

VCTree w/Bayesian head: `MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreeHierPredictor`

VTransE w/Bayesian head: `MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerHierPredictor`


For example, a complete training cmd for NeuralMotif w/Bayesian head should be:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifHierarchicalPredictor SOLVER.PRE_VAL False SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 80000 SOLVER.BASE_LR 0.0025 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 1000 GLOVE_DIR /raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/glove MODEL.PRETRAINED_DETECTOR_CKPT /raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/motif-hierarch
```

and the evaluation command will be: 

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifHierarchicalPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/glove MODEL.PRETRAINED_DETECTOR_CKPT /raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/motif-hierarch OUTPUT_DIR /raid0/docker-raid/bwjiang/scene_graph/checkpoints/benchmark/motif-hierarch
```