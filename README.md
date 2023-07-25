

# Awesome Prompting on Vision-Language Models

<img src="./assets/pvlm-mindmap.png" width="100%" height="100%">

## # :nerd_face: What is Prompting on Vision-Language Models?
Prompt engineering is a technique that involves augmenting a large pre-trained model with task-specific hints, known as prompts, to adapt the model to new tasks. This paper aims to provide **a comprehensive survey** of cutting-edge research in prompt engineering on **three** types of vision-language models (VLMs): **multimodal-to-text generation models** (*e.g.*, Flamingo), **image-text matching models** (*e.g.*, CLIP), and **text-to-image generation models** (*e.g.*, Stable Diffusion) (Fig. 1).

<img src="./assets/3-models.png">

<p align="center"> <i>Fig. 1 : Three main types of vision-language models focused on this work.</i> 		 </p>

### Reference

This repo lists relevant papers summarized in our survey: 

**A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models.** *Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, Philip Torr*. Preprint 2023. [[pdf]]()

If you find our paper and repo helpful to your research, please cite the following paper:
```latex
@article{gu2023survey,
  title={A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models},
  author={Gu, Jindong and Han, Zhen and Chen, Shuo, and Beirami, Ahmad and He, Bailan and Zhang, Gengyuan and Liao, Ruotong and Qin, Yao and Tresp, Volker and Torr, Philip}
  journal={TBD},
  year={2023}
}
```

## # :paperclips: Awesome Papers

### Prompting Model in Multimodal-to-Text Generation (*e.g.* on Flamingo)

There are two main types of fusion module approaches based on the integration of visual and textual modalities: **encoder-decoder as a multi-modal fusion module** and **decoder-only as a multi-modal fusion module**. Prompting methods can be divided into **two main categories**  (Fig. 2) based on the readability of the templates: **hard prompt** and **soft prompt**. Hard prompt encompasses four subcategories: *task instruction, in-context learning,* *retrieval-based prompting, and chain-of-thought prompting*. Soft prompts are classified into two strategies: *prompt tuning* and *prefix token tuning*, based on whether they internally add new tokens to the model's architecture or simply append them to the input. this study primarily concentrates on prompt methods that avoid altering the base model.

<img src="./assets/chapt3_prompting_method.png">

<p align="center">  <i>Fig. 2 : Classification of prompting methods.</i> 		 </p>



| Title                                                        | Venue       | Year | Code if available                                            | Comment                                              |
| :----------------------------------------------------------- | ----------- | ---- | ------------------------------------------------------------ | ---------------------------------------------------- |
| [Unifying Vision-and-Language Tasks via Text Generation](https://arxiv.org/abs/2102.02779) | ICML        | 2021 | [Github](https://github.com/j-min/VL-T5)                     | Encoder-decoder fusion; Text prefixes as prompt      |
| [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904) | ICLR        | 2022 | [Github](https://github.com/YulongBonjour/SimVLM)            | Encoder-decoder fusion; Text prefixes as prompt      |
| [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052) | ICML        | 2022 | [Github](https://github.com/OFA-Sys/OFA)                     | Encoder-decoder fusion; Text prefixes as prompt      |
| [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794) | ICLR        | 2023 | ---                                                          | Encoder-decoder fusion; Instruction prompt           |
| [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884) | NeurIPS     | 2021 | [Page](https://fh295.github.io/frozen.html)                  | Decoder-only fusion; Image conditional prefix tuning |
| [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | NeurIPS     | 2022 | [Github](https://github.com/mlfoundations/open_flamingo)     | Decoder-only fusion; Text prompts;                   |
| [MAGMA -- Multimodal Augmentation of Generative Models through Adapter-based Finetuning](https://aclanthology.org/2022.findings-emnlp.179/) | EMNLP       | 2022 | [Github](https://github.com/Aleph-Alpha/magma)               | Decoder-only fusion; Image conditional prefix tuning |
| [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) | ICML        | 2023 | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | Decoder-only fusion; Image conditional prefix tuning |
| [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) | OpenAI Blog | 2019 | [Github](https://github.com/openai/gpt-2)                    | Task instruction prompt                              |
| [The Turking Test: Can Language Models Understand Instructions?](https://arxiv.org/abs/2010.11982) | arXiv       | 2020 | ---                                                          | Task instruction prompt                              |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | NeurIPS     | 2020 | ---                                                          | In-context learning                                  |
| [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633) | NAACL-HLT   | 2022 | [Github](https://github.com/OhadRubin/EPR)                   | Retrieval-based prompting                            |
| [Unified Demonstration Retriever for In-Context Learning](https://arxiv.org/abs/2305.04320) | ACL         | 2023 | [Github](https://github.com/KaiLv69/UDR)                     | Retrieval-based prompting                            |
| [Compositional Exemplars for In-context Learning](https://arxiv.org/abs/2302.05698) | ICML        | 2023 | [Github](https://github.com/HKUNLP/icl-ceil)                 | Retrieval-based prompting                            |
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | NeurIPS     | 2022 | ---                                                          | Chain-of-thought prompting                           |
| [Automatic Chain of Thought Prompting in Large Language Models]() | ICLR        | 2023 | [Github](https://github.com/amazon-research/auto-cot)        | Chain-of-thought prompting                           |
| [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) | EMNLP       | 2021 | ---                                                          | Prompt tuning                                        |
| [Learning How to Ask: Querying LMs with Mixtures of Soft Prompts](https://arxiv.org/abs/2104.06599) | NAACL-HLT   | 2021 | [Github](https://github.com/hiaoxui/soft-prompts)            | Prompt tuning                                        |
| [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) | ACL         | 2021 | [Github](https://github.com/XiangLi1999/PrefixTuning)        | Prefix tuning                                        |
| [Prompt Tuning for Generative Multimodal Pretrained Models](https://arxiv.org/abs/2208.02532) | ACL         | 2023 | [Github](https://github.com/OFA-Sys/OFA)                     | Prompt tuning on OFA                                 |
| [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/abs/2302.14045) | arXiv       | 2023 | [Github](https://github.com/microsoft/unilm)                 | Textual instruction prompts                          |

### Applications & Responsible AI



## Prompting Model in Image-Text Matching (*e.g.* on CLIP)

Depending on the target of prompting, existing methods can be classified into three categories: **prompting the text encoder**, **prompting the visual encoder**, or **jointly prompting both branches** as shown in Fig. 2 . These approaches aim to enhance the flexibility and task-specific performance of VLMs.

<img src="./assets/chapt4_prompting_method.png">

<p align="center">  <i>Fig. 2 : Classification of prompting methods on Image-Text Matching VLMs. </i> 		 </p>



| Title                                                        | Venue   | Year | Code if available                                            | Comment                                            |
| ------------------------------------------------------------ | ------- | ---- | ------------------------------------------------------------ | -------------------------------------------------- |
| [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | ICML    | 2021 | [Github](https://github.com/OpenAI/CLIP)                     | Hard text prompts; Prompt for Image classification |
| [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=e8PVEkSa4Fq) | NeurIPS | 2022 | [Github](https://github.com/azshue/TPT)                      | Soft text prompts                                  |
| [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) | IJCV    | 2022 | [Github](https://github.com/KaiyangZhou/CoOp)                | Soft text prompts                                  |
| [Prompting Visual-Language Models for Efficient Video Understanding](https://arxiv.org/abs/2112.04478) | ECCV    | 2022 | [Github](https://github.com/ju-chen/Efficient-Prompt)        | Soft text prompts                                  |
| [Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720) | arXiv   | 2022 | [Github](https://github.com/sIncerass/MVLPT)                 | Soft text prompts                                  |
| [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557) | CVPR    | 2022 | [Github](https://github.com/KaiyangZhou/CoOp)                | Soft text prompts                                  |
| [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)     | ECCV    | 2022 | [Github](https://github.com/KMnP/vpt)                        | Visual patch-wise prompts                          |
| [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274) | arXiv   | 2022 | [Github](https://github.com/hjbahng/visual_prompting)        | Visual patch-wise prompts                          |
| [Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720) | arXiv   | 2022 | [Github](https://github.com/sIncerass/MVLPT)                 | Visual patch-wise prompts                          |
| [Unleashing the Power of Visual Prompting At the Pixel Level](https://arxiv.org/abs/2212.10556) | arXiv   | 2022 | [Github](https://github.com/UCSC-VLAA/EVP)                   | Visual patch-wise prompts                          |
| [Diversity-Aware Meta Visual Prompting](https://arxiv.org/abs/2303.08138) | CVPR    | 2023 | [Github](https://github.com/shikiw/DAM-VP)                   | Visual patch-wise prompts                          |
| [CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models](https://arxiv.org/abs/2109.11797) | arXiv   | 2022 | [Github](https://github.com/thunlp/CPT)                      | Visual annotation prompts                          |
| [What does CLIP know about a red circle? Visual prompt engineering for VLMs](https://arxiv.org/abs/2304.06712) | arXiv   | 2023 | ---                                                          | Visual annotation prompts                          |
| [Visual Prompting via Image Inpainting](https://arxiv.org/abs/2209.00647) | NeurIPS | 2022 | [Github](https://github.com/amirbar/visual_prompting)        | Visual annotation prompts                          |
| [Unified Vision and Language Prompt Learning](https://arxiv.org/abs/2210.07225) | arXiv   | 2023 | [Github](https://github.com/yuhangzang/UPT)                  | Coupled unified prompting                          |
| [Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720) | arXiv   | 2022 | [Github](https://github.com/sIncerass/MVLPT)                 | Decoupled unified prompting                        |
| [MaPLe: Multi-modal Prompt Learning](https://arxiv.org/abs/2210.03117) | CVPR    | 2023 | [Github](https://github.com/muzairkhattak/multimodal-prompt-learning) | Decoupled unified prompting                        |
| [Understanding Zero-shot Adversarial Robustness for Large-Scale Models](https://openreview.net/forum?id=P4bXCawRi5J) | ICLR    | 2023 | [Code](https://www.catalyzex.com/paper/arxiv:2212.07016/code) | Adversarial robustness of prompt                   |
| [Visual Prompting for Adversarial Robustness](https://arxiv.org/abs/2210.06284) | ICASSP  | 2023 | [Github](https://github.com/Phoveran/vp-for-adversarial-robustness) | Adversarial robustness of prompt                   |
| [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651) | NeurIPS | 2021 | [Github](https://github.com/salesforce/ALBEF/)               | Image-Text Matching Model                          |
| [Unsupervised Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2204.03649) | arXiv   | 2022 | [Github](https://github.com/tonyhuang2022/UPL)               | Unspervised learnable prompts                      |
| [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://arxiv.org/abs/2209.07511) | NeurIPS | 2022 | [Github](https://github.com/azshue/TPT)                      | Learnable prompt                                   |

### Applications & Responsible AI

| Title                                                        | Venue         | Year | Code if available                                            | Comment                                                      |
| ------------------------------------------------------------ | ------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://arxiv.org/abs/2209.07511) | NeurIPS       | 2022 | [Github](https://github.com/azshue/TPT)                      | Learnable prompt; Prompts for image classification           |
| [LPT: Long-tailed Prompt Tuning for Image Classification](https://openreview.net/forum?id=8pOVAeo8ie) | ICLR          | 2023 | [Github](https://github.com/DongSky/LPT)                     | Prompts for long-tailed image classification                 |
| [Texts as Images in Prompt Tuning for Multi-Label Image Recognition](https://arxiv.org/abs/2211.12739) | CVPR          | 2023 | [Github](https://github.com/guozix/TaI-DPT)                  | Prompts for multi-label image classification and detection   |
| [DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations](https://arxiv.org/abs/2206.09541) | NeurIPS       | 2022 | [Github](https://github.com/sunxm2357/DualCoOp)              | Prompts for multi-label image classification and recognition |
| [Visual Prompt Tuning for Few-Shot Text Classification](https://aclanthology.org/2022.coling-1.492.pdf) | ICCL          | 2022 | ---                                                          | Visual prompts for text classification                       |
| [Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921) | ICLR          | 2021 | [Github](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) | Prompts for object detection                                 |
| [Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model](https://arxiv.org/abs/2203.14940) | CVPR          | 2022 | [Github](https://github.com/dyabel/detpro)                   | Prompts for object detection                                 |
| [PromptDet: Towards Open-vocabulary Detection using Uncurated Images](https://arxiv.org/abs/2203.16513) | ECCV          | 2022 | [Github](https://fcjian.github.io/promptdet)                 | Prompts for object detection                                 |
| [Optimizing Continuous Prompts for Visual Relationship Detection by Affix-Tuning](https://ieeexplore.ieee.org/document/9815128) | IEEE Access   | 2022 | ---                                                          | Soft prompts for visual relation detection                   |
| [Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning](https://arxiv.org/abs/2208.08165) | ECCV          | 2022 | ---                                                          | Soft prompts for visual relation detection                   |
| [Compositional Prompt Tuning with Motion Cues for Open-vocabulary Video Relation Detection](https://arxiv.org/abs/2302.00268) | ICLR          | 2023 | [Github](https://github.com/Dawn-LX/OpenVoc-VidVRD)          | Relation Prompts for video open-vocabulary relation detection |
| [DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting](https://arxiv.org/abs/2112.01518) | CVPR          | 2022 | [Github](https://github.com/raoyongming/DenseCLIP)           | Class-conditioned text prompts for semantic segmentation     |
| [Segment Anything](https://arxiv.org/abs/2304.02643)         | ICCV          | 2023 | [Github](https://segment-anything.com/)                      | Promptable queries for semantic segmentation                 |
| [Domain Adaptation via Prompt Learning](https://arxiv.org/abs/2202.06687) | arXiv         | 2022 | [Github](https://github.com/LeapLabTHU/DAPrompt)             | Domain-specific textual prompts for domain adaptation        |
| [Visual Prompt Tuning for Test-time Domain Adaptation]()     | arXiv         | 2022 | ---                                                          | Prompts for domain adaptation                                |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | CVPR          | 2022 | [Github](https://github.com/google-research/l2p)             | Prompts for continual learning                               |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | ECCV          | 2022 | [Github](https://github.com/google-research/l2p)             | Prompts for continual learning                               |
| [Prompt Vision Transformer for Domain Generalization]()      | arXiv         | 2022 | [Github](https://github.com/zhengzangw/DoPrompt)             | Prompts for domain generalization                            |
| [Understanding Zero-Shot Adversarial Robustness for Large-Scale Models](https://arxiv.org/abs/2212.07016) | arXiv         | 2022 | [Github](https://github.com/cvlab-columbia/ZSRobust4FoundationModel) | Visual prompt tuning under adversarial attack                |
| [Visual Prompting for Adversarial Robustness](https://arxiv.org/abs/2210.06284) | ICASSP        | 2023 | [Github](https://github.com/Phoveran/vp-for-adversarial-robustness) | Visual prompting to improve the adversarial robustness       |
| [Exploring the Universal Vulnerability of Prompt-based Learning Paradigm](https://arxiv.org/abs/2204.05239) | NAACL         | 2022 | [Github](https://github.com/leix28/prompt-universal-vulnerability) | Visual prompting vulnerability                               |
| [Poisoning and Backdooring Contrastive Learning](https://openreview.net/forum?id=iC4UHbQ01Mp) | ICLR          | 2022 | ---                                                          | Backdoor and poisoning attacks on CLIP                       |
| [BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning](https://ieeexplore.ieee.org/abstract/document/9833644) | IEEE          | 2022 | [Github](https://github.com/jjy1994/BadEncoder)              | Backdoor attack on CLIP                                      |
| [CleanCLIP: Mitigating Data Poisoning Attacks in Multimodal Contrastive Learning ](https://openreview.net/forum?id=GfgCNeVRFhV) | ICLR Workshop | 2023 | ---                                                          | Defense backdoor attacks on CLIP                             |
| [Debiasing Vision-Language Models via Biased Prompts](https://arxiv.org/abs/2302.00070) | arXiv         | 2023 | [Github](https://github.com/chingyaoc/debias_vl)             | Prompts to alleviate bias                                    |



## Prompting Model in Text-to-Image Generation (*e.g.* on Stable Diffusion)

### Text-to-Image Generation Overview 

| Title                                                        | Venue   | Year | Code if available                                    | Comment                              |
| ------------------------------------------------------------ | ------- | ---- | ---------------------------------------------------- | ------------------------------------ |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | NeurIPS | 2021 | [Github](https://github.com/openai/guided-diffusion) | Diffusion models on image generation |
|                                                              |         |      |                                                      |                                      |
|                                                              |         |      |                                                      |                                      |



### Prompting Methods 

| Title                                                        | Venue            | Year | Code if available                                            | Comment                              |
| ------------------------------------------------------------ | ---------------- | ---- | ------------------------------------------------------------ | ------------------------------------ |
| [Investigating Prompt Engineering in Diffusion Models](https://arxiv.org/abs/2211.15462) | NeurIPS Workshop | 2022 | ---                                                          | Semantic prompt design               |
| [DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models](https://arxiv.org/abs/2303.11681) | arXiv            | 2023 | [Github](https://github.com/weijiawu/DiffuMask)              | Diversify generation with prompt     |
| [Is synthetic data from generative models ready for image recognition?](https://arxiv.org/abs/2210.07574) | ICLR             | 2023 | [Github](https://github.com/CVMI-Lab/SyntheticData)          | Diversify generation with prompt     |
| [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618) | ICLR             | 2023 | [Github](https://textual-inversion.github.io/)               | Complex control of synthesis results |
| [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) | CVPR             | 2023 | [Github](https://github.com/google/dreambooth)               | Complex control of synthesis results |
| [Multi-Concept Customization of Text-to-Image Diffusion](https://arxiv.org/abs/2212.04488) | CVPR             | 2023 | [Github](https://github.com/adobe-research/custom-diffusion) | Complex control of synthesis results |
| [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626) | arXiv            | 2022 | ---                                                          | Complex control of synthesis results |
| TBD applications & ethics papers                             |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |



## Prompting VLMs vs. Uni-modal Models

TBD 



## # :mailbox_with_mail: Contact 

Please contact us (jindong.gu@outlook.com, chenshuo.cs@outlook.com) if 
- you would like to add your paper in this repo,
- you find any mistake in this repo, 
- you have any suggestion for this repo. 

