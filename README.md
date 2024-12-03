# DietCoke
This is the code related to "Diversify, Rationalize, and Combine: Ensembling Multiple QA Strategies for Zero-shot Knowledge-based VQA" (Findings of EMNLP'24).
![](https://github.com/limiaoyu/REACT/blob/main/DietCoke.jpg)


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

## Citation

If you find it helpful to your research, please cite as follows:
```
@inproceedings{li-etal-2024-diversify,
    title = "Diversify, Rationalize, and Combine: Ensembling Multiple {QA} Strategies for Zero-shot Knowledge-based {VQA}",
    author = "Li, Miaoyu  and
      Li, Haoxin  and
      Du, Zilin  and
      Li, Boyang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    pages = "1552--1566"
}
```

## Acknowledgements
Note that this code is based on the [Img2llm](https://github.com/CR-Gjx/Img2Prompt) repo.

