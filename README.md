# SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models

This is the official implementation of NeurIPS 2024 paper "[SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models](https://arxiv.org/abs/2403.07384)".

Authors: [Yu Yang](https://sites.google.com/g.ucla.edu/yuyang/home), Siddhartha Mishra, [Jeffrey N Chiang](https://scholar.google.com/citations?user=4Hb-E48AAAAJ&hl=en), [Baharan Mirzasoleiman](https://baharanm.github.io/)

## Overview

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## ⚙️ Installation

1. Clone the repository.

2. Follow the [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) to create a new conda environment and install vllm. Make sure to check the CUDA version and install the corresponding version of vllm.

3. Install the following required packages.

```bash
pip install accelerate wandb
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
```

## 🔧 Usage

### Data Selection with S2L

#### 1. Fine-tune the small proxy model.

- **If you have your own training script**, please go ahead and use it to train the small proxy model and save the desired number of model checkpoints during training.

- **If you want to reproduce our experiments**, you can run the following command, using the example configuration file `configs/pythia-70m-deduped_checkpoints.yml` to train a Pythia-70M model on the [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) dataset.

    ```bash
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT train.py --config_file configs/pythia-70m-deduped_checkpoints.yml --wandb_key $WANDB_KEY
    ```

    The training script will save the model checkpoints to the `res/full_mathinstruct_pythia-70m-deduped_3epochs_512_checkpoints` directory.

#### 2. Collect the training trajectories of the small model.

If you used the training script provided in this repo, you can collect the training trajectories of the small model. We provide two methods:

```bash
# Process all checkpoints found in the model directory:
python run_distributed_trajectories.py --model_path res/full_mathinstruct_pythia-70m-deduped_3epochs_512_checkpoints --config_file configs/pythia-70m-deduped_checkpoints.yml --checkpoints all

# Or specify specific checkpoints:
# Comma-separated list:
python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints 1000,2000,3000,4000

# Using a range (start:end:step):
python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints 1000:5000:1000
```

The distributed script will automatically detect available GPUs and distribute the checkpoint processing across them for faster computation.

This script requires the configuration file used for training the small model. 

If you want to compute the loss for a specific checkpoint, you can run the following command with the checkpoint path by `--ckpt`.

```bash
python get_trajectories.py --config_file configs/pythia-70m-deduped_checkpoints.yml --ckpt 1000
```

#### 3. Use S2L to select data for finetuning the large model. 

```bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT train.py --config_file configs/s2l/full-70m_100_phi-3-mini-4k-instruct_130k_3epochs.yml --wandb_key $WANDB_KEY
```


## 📄 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
    yang2024smalltolarge,
    title={SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models},
    author={Yu Yang and Siddhartha Mishra and Jeffrey N Chiang and Baharan Mirzasoleiman},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=K9IGlMQpif}
}
```

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions or suggestions, please contact [Yu Yang](mailto:yuyang@cs.ucla.edu). 🤗
