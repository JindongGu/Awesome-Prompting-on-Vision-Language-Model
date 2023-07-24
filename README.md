

# Awesome Prompting on Vision-Language Models

<img src="./assets/pvlm-mindmap.png" width="100%" height="100%">

## # :nerd_face: What is Prompting on Vision-Language Models?
Prompt engineering is a technique that involves augmenting a large pre-trained model with task-specific hints, known as prompts, to adapt the model to new tasks. This paper aims to provide **a comprehensive survey** of cutting-edge research in prompt engineering on **three** types of vision-language models (VLMs): **multimodal-to-text generation models** (*e.g.*, Flamingo), **image-text matching models** (*e.g.*, CLIP), and **text-to-image generation models** (*e.g.*, Stable Diffusion) (Fig. 1).

<img src="./assets/3-models.png">

<p align="center"> <i>Fig. 1 : Three main types of vision-language models focused on this work.</i> 		 </p>

### Reference

This repo lists relevant papers summarized in our survey: 

**A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models.** *Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, Philip Torr*. Preprint 2023. [[pdf]](https://scholar.google.com/citations?user=mj3ff80AAAAJ&hl=en)

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



| Title                                                        | Venue   | Year | Code if available                                            | Comment                |
| :----------------------------------------------------------- | ------- | ---- | ------------------------------------------------------------ | ---------------------- |
| [Unifying Vision-and-Language Tasks via Text Generation](https://arxiv.org/abs/2102.02779) | ICML    | 2021 | [Github](https://github.com/j-min/VL-T5)                     | Encoder-decoder fusion |
| [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904) | ICLR    | 2022 | [Github](https://github.com/YulongBonjour/SimVLM)            | Encoder-decoder fusion |
| [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052) | ICML    | 2022 | [Github](https://github.com/OFA-Sys/OFA)                     | Encoder-decoder fusion |
| [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794) | ICLR    | 2023 | --                                                           | Encoder-decoder fusion |
| [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884) | NeurIPS | 2021 | [Page](https://fh295.github.io/frozen.html)                  | Decoder-only fusion    |
| [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | NeurIPS | 2022 | [Github](https://github.com/mlfoundations/open_flamingo)     | Decoder-only fusion    |
| [MAGMA -- Multimodal Augmentation of Generative Models through Adapter-based Finetuning](https://aclanthology.org/2022.findings-emnlp.179/) | EMNLP   | 2022 | [Github](https://github.com/Aleph-Alpha/magma)               | Decoder-only fusion    |
| [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) | ICML    | 2023 | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | Decoder-only fusion    |
|                                                              |         |      |                                                              |                        |

## Prompting Model in Image-Text Matching (*e.g.* on CLIP)

Depending on the target of prompting, existing methods can be classified into three categories: **prompting the text encoder**, **prompting the visual encoder**, or **jointly prompting both branches** as shown in Fig. 2 . These approaches aim to enhance the flexibility and task-specific performance of VLMs.

<img src="./assets/chapt4_prompting_method.png">

<p align="center">  <i>Fig. 2 : Classification of prompting methods on Image-Text Matching VLMs. </i> 		 </p>



| Title                                                        | Venue   | Year | Code if available                                            | Comment                          |
| ------------------------------------------------------------ | ------- | ---- | ------------------------------------------------------------ | -------------------------------- |
| [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | ICML    | 2021 | [Github](https://github.com/OpenAI/CLIP)                     | Hard text prompts                |
| [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=e8PVEkSa4Fq) | NeurIPS | 2022 | [Github](https://github.com/azshue/TPT)                      | Soft text prompts                |
| [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) | IJCV    | 2022 | [Github](https://github.com/KaiyangZhou/CoOp)                | Soft text prompts                |
| [Prompting Visual-Language Models for Efficient Video Understanding](https://arxiv.org/abs/2112.04478) | ECCV    | 2022 | [Github](https://github.com/ju-chen/Efficient-Prompt)        | Soft text prompts                |
| [Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720) | arXiv   | 2022 | [Github](https://github.com/sIncerass/MVLPT)                 | Soft text prompts                |
| [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557) | CVPR    | 2022 | [Github](https://github.com/KaiyangZhou/CoOp)                | Soft text prompts                |
| [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)     | ECCV    | 2022 | [Github](https://github.com/KMnP/vpt)                        | Visual patch-wise prompts        |
| [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274) | arXiv   | 2022 | [Github](https://github.com/hjbahng/visual_prompting)        | Visual patch-wise prompts        |
| [Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720) | arXiv   | 2022 | [Github](https://github.com/sIncerass/MVLPT)                 | Visual patch-wise prompts        |
| [Unleashing the Power of Visual Prompting At the Pixel Level](https://arxiv.org/abs/2212.10556) | arXiv   | 2022 | [Github](https://github.com/UCSC-VLAA/EVP)                   | Visual patch-wise prompts        |
| [Diversity-Aware Meta Visual Prompting](https://arxiv.org/abs/2303.08138) | CVPR    | 2023 | [Github](https://github.com/shikiw/DAM-VP)                   | Visual patch-wise prompts        |
| [CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models](https://arxiv.org/abs/2109.11797) | arXiv   | 2022 | [Github](https://github.com/thunlp/CPT)                      | Visual annotation prompts        |
| [What does CLIP know about a red circle? Visual prompt engineering for VLMs](https://arxiv.org/abs/2304.06712) | arXiv   | 2023 | -                                                            | Visual annotation prompts        |
| [Visual Prompting via Image Inpainting](https://arxiv.org/abs/2209.00647) | NeurIPS | 2022 | [Github](https://github.com/amirbar/visual_prompting)        | Visual annotation prompts        |
| [Unified Vision and Language Prompt Learning](https://arxiv.org/abs/2210.07225) | arXiv   | 2023 | [Github](https://github.com/yuhangzang/UPT)                  | Coupled unified prompting        |
| [Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720) | arXiv   | 2022 | [Github](https://github.com/sIncerass/MVLPT)                 | Decoupled unified prompting      |
| [MaPLe: Multi-modal Prompt Learning](https://arxiv.org/abs/2210.03117) | CVPR    | 2023 | [Github](https://github.com/muzairkhattak/multimodal-prompt-learning) | Decoupled unified prompting      |
| [Understanding Zero-shot Adversarial Robustness for Large-Scale Models](https://openreview.net/forum?id=P4bXCawRi5J) | ICLR    | 2023 | [Code](https://www.catalyzex.com/paper/arxiv:2212.07016/code) | Adversarial robustness of prompt |
| [Visual Prompting for Adversarial Robustness](https://arxiv.org/abs/2210.06284) | ICASSP  | 2023 | [Github](https://github.com/Phoveran/vp-for-adversarial-robustness) | Adversarial robustness of prompt |



## Prompting Model in Text-to-Image Generation (*e.g.* on Stable Diffusion)

| Title                                                        | Venue            | Year | Code if available                                            | Comment                              |
| ------------------------------------------------------------ | ---------------- | ---- | ------------------------------------------------------------ | ------------------------------------ |
| [Investigating Prompt Engineering in Diffusion Models](https://arxiv.org/abs/2211.15462) | NeurIPS Workshop | 2022 | ---                                                          | Semantic prompt design               |
| [DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models](https://arxiv.org/abs/2303.11681) | arXiv            | 2023 | [Github](https://github.com/weijiawu/DiffuMask)              | Diversify generation with prompt     |
| [Is synthetic data from generative models ready for image recognition?](https://arxiv.org/abs/2210.07574) | ICLR             | 2023 | [Github](https://github.com/CVMI-Lab/SyntheticData)          | Diversify generation with prompt     |
| [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618) | ICLR             | 2023 | [Github](https://textual-inversion.github.io/)               | Complex control of synthesis results |
| [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) | CVPR             | 2023 | [Github](https://github.com/google/dreambooth)               | Complex control of synthesis results |
| [Multi-Concept Customization of Text-to-Image Diffusion](https://arxiv.org/abs/2212.04488) | CVPR             | 2023 | [Github](https://github.com/adobe-research/custom-diffusion) | Complex control of synthesis results |
| [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626) | arXiv            | 2022 | --                                                           | Complex control of synthesis results |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |
|                                                              |                  |      |                                                              |                                      |



## Prompting VLMs vs. Uni-modal Models 



## # :mailbox_with_mail: Contact 

Please contact us (jindong.gu@outlook.com, chenshuo.cs@outlook.com) if 
- you would like to add your paper in this repo,
- you find any mistake in this repo, 
- you have any suggestion for this repo. 

