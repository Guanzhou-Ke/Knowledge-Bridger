# Knowledge-Bridger

Official repository for the paper "[Knowledge Bridger: Towards Training-Free Missing Modality Completion](https://arxiv.org/pdf/2502.19834)".

[[📖 Paper](https://arxiv.org/pdf/2502.19834)]

- This paper is accepted at **CVPR 2025** 🎉.


## TLDR;

We explore the potential of leveraging large multimodal models to generate missing modalities. In scenarios with limited computing resources (such as scarce data and GPUs), fine-tuning an end-to-end model becomes challenging. To address this, we aim to harness large models to extract cross-modal knowledge, which in turn guides pretrained generative models in producing the desired missing data. Unlike traditional generative paradigms, our approach first derives a refined prompt based on the extracted knowledge. This new prompt then directs the generative model to synthesize the missing data. Finally, we employ ranking tools to identify the most plausible missing data by evaluating semantic similarity and knowledge graph similarity between the generated modality and the available modalities.


## Abstract

Previous successful approaches to missing modality completion rely on carefully designed fusion techniques and extensive pre-training on complete data, which can limit their generalizability in out-of-domain (OOD) scenarios. In this study, we pose a new challenge: **can we develop a missing modality completion model that is both resource-efficient and robust to OOD generalization?** To address this, we present a training-free framework for missing modality completion that leverages large multimodal model (LMM). Our approach, termed the ``Knowledge Bridger'', is modality-agnostic and integrates generation and ranking of missing modalities. By defining domain-specific priors, our method automatically extracts structured information from available modalities to construct knowledge graphs. These extracted graphs connect the missing modality generation and ranking modules through the LMM, resulting in high-quality imputations of missing modalities. Experimental results across both general and medical domains show that our approach consistently outperforms competing methods, including in OOD generalization. Additionally, our knowledge-driven generation and ranking techniques demonstrate superiority over variants that directly employ LMMs for generation and ranking, offering insights that may be valuable for applications in other domains.

The workflow of this work.

<p align="center">
    <img src="misc/framework.jpg" width="100%"> <br>
</p>


## Get Started

### Installation

1. This project is based on Python 3.11 and 
