# DGCDR

Pytorch implementation for paper "**Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement**"[[1]](#1). The paper has been accepted for presentation at the [RecSys 2025 conference](https://recsys.acm.org/recsys25/accepted-contributions/). Thank you for your support and citation!

## Datasets
The dataset is processed based on [Amazon](https://amazon-reviews-2023.github.io/index.html)[[2]](#2), Douban[[3]](#3). 

Each domain contains a interaction file in the following format:

| Suffix | Content | Format |
|---|---|---|
| *.inter* | User-item interaction | user_id, item_id, rating |

## Training
You can use this command to train the model:

`python run_recbole_cdr.py`

The configuration settings are in a YAML file in  DGCDR/recbole_cdr/properties/.

## Acknowledgement
The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR).


## References
## References
<a id="1">[1]</a> Yuhan Wang, Qing Xie, Zhifeng Bao, Mengzi Tang, Lin Li, and Yongjian Liu. 2025. Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement. In Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys ’25). Association for Computing Machinery, New York, NY, USA, 104–113. https://doi.org/10.1145/3705328.3748044

<a id="2">[2]</a> Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2024. Bridging Language and Items for Retrieval and Recommendation. arXiv preprint arXiv:2403.03952 (2024).

<a id="3">[3]</a> Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu
Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, et al., Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms,in: Proceedings of the 30th ACM International Conference on Information & Knowledge Management, Association for Computing Machinery, New York, NY, USA, 2021, pp. 4653–4664


