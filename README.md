# REACT
This is the code related to "Rationale-based Ensemble of Multiple QA Strategies for Zero-shot Knowledge-based VQA" (Findings of EMNLP24).
![](https://github.com/limiaoyu/REACT/blob/main/overview.jpg)


## Usage
### Step1: Generate Knowledge
You can generate the short-form knowledge and long-form knowledge with:
```
$ python VL_captioning/knowledge_generate.py  
```
We provide the results in [VL_captioning/results/short_knowledge.json](https://github.com/limiaoyu/REACT/blob/main/VL_captioning/results/short_knowledge.json)
### Testing
You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training (the best valiteration for 2D and 3D is shown at the end of each training). Note that @ will be replaced by the output directory for that config file. For example:
```
$ python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda.yaml  @/model_2d_065000.pth @/model_3d_095000.pth
```
You can also provide an absolute path without `@`. 

## Paper
[Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation](https://dl.acm.org/doi/10.1145/3503161.3547990)

**MM '22: Proceedings of the 30th ACM International Conference on Multimedia**

If you find it helpful to your research, please cite as follows:
```
@inproceedings{li2022cross,
  title={Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation},
  author={Li, Miaoyu and Zhang, Yachao and Xie, Yuan and Gao, Zuodong and Li, Cuihua and Zhang, Zhizhong and Qu, Yanyun},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3829--3837},
  year={2022}
}
```

## Acknowledgements
Note that this code is based on the [Img2llm](https://github.com/CR-Gjx/Img2Prompt) repo.

