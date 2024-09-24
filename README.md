# REACT
This is the code related to "Rationale-based Ensemble of Multiple QA Strategies for Zero-shot Knowledge-based VQA" (Findings of EMNLP24).
![](https://github.com/limiaoyu/REACT/blob/main/overview.jpg)


## Usage
### Step1: Generate Knowledge
You can generate the short-form knowledge and long-form knowledge with:
```
$ python VL_captioning/knowledge_generate.py  
```
We provide the results in [short/long_knowledge.json](https://github.com/limiaoyu/REACT/tree/main/VL_captioning/results)

### Step2: Generate Answer Candidate
You can generate the answer candidate with:
```
$ python VL_captioning/candidate_generate.py --knowledge_file=the/path/to/knowledge.json
```
We provide the results in [candidate_cap/short/long.json](https://github.com/limiaoyu/REACT/tree/main/VL_captioning/results)

### Step3: Generate Rationales
You can generate the automatic rationale with:
```
$ python VL_captioning/auto_rationale_generate.py --answer_file=the/path/to/candidate.json
```
We provide the results in [AR_cap/short/long.json](https://github.com/limiaoyu/REACT/tree/main/VL_captioning/results)

You can generate the mechanistic rationale with:
```
$ python VL_captioning/mech_rationale_generate.py --knowledge_file=the/path/to/knowledge.json
```
We provide the results in [MR_cap/long.json](https://github.com/limiaoyu/REACT/tree/main/VL_captioning/results)

Notice that the MR_short.json is short_knowledge.json.

### Step4: Answer Fusion
You can fuse the answer candidates with:
```
$ python VL_captioning/answer_fusion.py 
```
We provide the result in [final_ans.json](https://github.com/limiaoyu/REACT/tree/main/VL_captioning/results)

## Paper

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

