# TAG: a Tuning-free Attention-driven GUI Grounding method for GUI task automation

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2412.10840">Attention-driven GUI Grounding: Leveraging Pretrained Multimodal Large Language Models without Fine-Tuning</a>
</p>

<div>
  <p align="center" style="font-size: larger;">
    <strong>AAAI 2025</strong> 
  </p>
</div>

**Abstract:** Recent advancements in Multimodal Large Language Models (MLLMs) have generated significant interest in their ability to autonomously interact with and interpret Graphical User Interfaces (GUIs). A major challenge in these systems is grounding—accurately identifying critical GUI components such as text or icons based on a GUI image and a corresponding text query. Traditionally, this task has relied on fine-tuning MLLMs with specialized training data to predict component locations directly. However, in this paper, we propose a novel Tuning-free Attention-driven Grounding (TAG) method that leverages the inherent attention patterns in pretrained MLLMs to accomplish this task without the need for additional fine-tuning. Our method involves identifying and aggregating attention maps from specific tokens within a carefully constructed query prompt. Applied to MiniCPM-Llama3-V 2.5, a state-of-the-art MLLM, our tuning-free approach achieves performance comparable to tuning-based methods, with notable success in text localization. Additionally, we demonstrate that our attention map-based grounding technique significantly outperforms direct localization predictions from MiniCPM-Llama3-V 2.5, highlighting the potential of using attention maps from pretrained MLLMs and paving the way for future innovations in this domain.</div>

<p align="center">
<img src="https://github.com/user-attachments/assets/2bac2e00-ac3d-493b-ab48-647a2e55e0e6" width=90%>
<p>
<be>

## Evaluation
### Task1: Optical Character Grounding
In this work, we developed an OCG dataset to evaluate the optical character grounding ability of MLLMs. Based on common screen resolutions, we construct 10 different (width:height) aspect ratios to comprehensively assess model's grounding robustness.
<img width="1125" alt="image" src="https://github.com/user-attachments/assets/64ed7323-a59f-42d2-b29b-786fe7c26676" />

### Task2: GUI Element Grounding
We evaluate methods on the [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) dataset.
<img width="1265" alt="image" src="https://github.com/user-attachments/assets/9980770c-c035-474d-bde1-549b277099ab" />

**Visual Demonstration:**
<img width="1341" alt="image" src="https://github.com/user-attachments/assets/f1fe3202-505b-49c0-8111-f18749d6a0e3" />

### Task3: GUI Agent Evaluation
We evaluate methods on the Element accuracy metric on [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) dataset.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/bb6f515e-0f02-46a8-8540-76232c4f674c" />

**Visual Demonstration:**
<img width="1340" alt="image" src="https://github.com/user-attachments/assets/24bda285-ceef-4c19-b2e6-bbed171d78c0" />

## Installation
### Dataset Preparation

Please download the following three benchmarks: [OCG](https://drive.google.com/drive/folders/1cIwpxUH6NlZ2JX1AKxLa44ule9cS6jOA?usp=sharing), [ScreenSpot](https://drive.google.com/drive/folders/1FuFT05yXOV_QxhwYft85YTLOgaIYm_fS) and Mind2Web [screenshots](https://box.nju.edu.cn/f/33e203d170ab48b0b922/) and [annotations](https://box.nju.edu.cn/f/e30b861fa7604668821b/). Please **DO NOT** redistribute the unzipped data files online to avoid risks such as model overfitting.

### Env Preparation
1. Clone this repository and enter the project folder

```bash
git clone https://github.com/HeimingX/TAG.git
cd TAG
```

2. Create conda environment

```Shell
conda create -n TAG python=3.10 -y
conda activate TAG
```

3. Install dependencies

```shell
pip install -r requirements.txt
```

## Eval Scripts
### Task1: OCG

```shell
MLLM_PATH=openbmb/MiniCPM-Llama3-V-2_5
OCG_DATAPATH=PATH/TO/OCG
IMG_ASPECTS='[[1, 4], [9, 21], [9, 19], [1, 2], [9, 16], [4, 3], [16, 9], [2, 1], [21, 9], [4, 1]]'

# Evaluate with MiniCPMV2.5
python eval_mm/ocg/eval_MiniCPMV2_5.py \
    --mllm_path ${MLLM_PATH} \
    --data_path ${OCG_DATAPATH} \
    --image_aspects "${IMG_ASPECTS}" \
    --save-dir output/ocg/mv2_5 \
    --sampling

# Evaluate with TAG
python eval_mm/ocg/eval_TAG.py \
    --mllm_path ${MLLM_PATH} \
    --data_path ${OCG_DATAPATH} \
    --image_aspects "${IMG_ASPECTS}" \
    --save-dir output/ocg/tag \
    --batchsize 4
```

### Task2: ScreenSpot
```shell
MLLM_PATH=openbmb/MiniCPM-Llama3-V-2_5
SCREENSPOT_IMGS=PATH/TO/IMGS
SCREENSPOT_TEST=PATH/TO/TESTSET

# Evaluate with MiniCPMV2.5
python eval_mm/screenspot/eval_MiniCPMV2_5.py \
    --mllm_path ${MLLM_PATH} \
    --screenspot_imgs ${SCREENSPOT_IMGS} \
    --screenspot_test ${SCREENSPOT_TEST} \
    --save-dir output/screenspot/mv2_5

# Evaluate with TAG
python eval_mm/screenspot/eval_TAG.py \
    --mllm_path ${MLLM_PATH} \
    --screenspot_imgs ${SCREENSPOT_IMGS} \
    --screenspot_test ${SCREENSPOT_TEST} \
    --save-dir output/screenspot/tag
```

### Task3: Mind2Web
```shell
MLLM_PATH=openbmb/MiniCPM-Llama3-V-2_5
MIND2WEB_DATAPATH=PATH/TO/MIND2WEB

TASKTYPES=(task website domain)
for TASK in "${TASKTYPES[@]}"
do
    # Evaluate with MiniCPMV2.5
    python eval_mm/mind2web/eval_MiniCPMV2_5.py \
        --mllm_path ${MLLM_PATH} \
        --data_dir ${MIND2WEB_DATAPATH} \
        --task ${TASK} \
        --save-dir output/mind2web/mv2_5

    # Evaluate with TAG
    python eval_mm/mind2web/eval_TAG.py \
        --mllm_path ${MLLM_PATH} \
        --data_dir ${MIND2WEB_DATAPATH} \
        --task ${TASK} \
        --save-dir output/mind2web/tag \
done
```

### Single Inference
Run a single image/query pair with optional tokenwise visualization:

```shell
python single_inference.py \
    --mllm_path ${MLLM_PATH} \
    --image path/to/image.jpg \
    --query "target text" \
    --save-dir output/demo \
    --vis-flag --tokenwise-vis
```

> Note: some evaluation [log](https://drive.google.com/drive/folders/1cIwpxUH6NlZ2JX1AKxLa44ule9cS6jOA?usp=sharing) files are provided for reference.

## Acknowledgement

We thank the following [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V/), [SeeClick](https://github.com/njucckevin/SeeClick) and [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) for their impressive work and open-sourced projects.


## Citation
If you find our code/paper helpful, please consider cite our paper 📝 and star us ⭐️！

```bibtext
@inproceedings{xu2025tag,
    title={Attention-driven GUI Grounding: Leveraging Pretrained Multimodal Large Language Models without Fine-Tuning},
    author={Hai-Ming Xu and Qi Chen and Lei Wang and Lingqiao Liu},
    booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
    year={2025},
}
```
