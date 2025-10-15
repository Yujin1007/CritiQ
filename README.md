
# Run training 
in CritiQ, 
./run_push_critiq.sh push_task/config/train_config.json 
./test_push_critiq.sh push_task/config/test_config.json 

# Download buffer


# How to install 
conda create -n critiq python=3.9.19 pip -y
conda activate critiq
python -m pip install -r requirements.txt

# Distilling Realizable Students from Unrealizable Teachers

Official implementation of CritiQ in pushing task-- a policy distillation under information assymetry


by [Yujin Kim*](https://Yujin1007.github.io/), [Nathaniel Chin*](https://www.linkedin.com/in/nathaniel-chin-5b2301195/), [Arnav Vasudev](https://www.linkedin.com/in/arnav-vasudev-a5a0811b2/),and [Sanjiban Choudhury &dagger;](https://sanjibanc.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05294-df2a2a.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2505.09546)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
[![Website](https://img.shields.io/badge/🔗-WebSite-black?style=for-the-badge)](https://portal-cornell.github.io/CritiQ_ReTRy/)



## Setup :hammer_and_wrench:

Create a virtual environment and activate it
```bash
# Conda
conda create -n critiq python=3.9.19 pip -y
conda activate critiq
git clone https://github.com/Yujin1007/CritiQ.git
cd CritiQ
python -m pip install -r requirements.txt
```
You could download buffer to jumpstart training
```
pip install gdown
gdown "https://drive.google.com/file/d/1Zp6rZ2oKqkruvoAlegvdAVeKY0gUMFfX/view?usp=sharing" -O ./push_task/buffer/buffer_1k.pkl
```
If you would prefer start training from collecting teacher demonstration, change config file in `CritiQ/push_task/config/train_config.json` into
```
{
...
  "buffer_path": "n",
...
  "collect_data": true,
  ...
}
```
Detailed descritions about configuration arguments can be found in `CritiQ/push_task/config/config.py`
## Training :robot:
```
./run_push_critiq.sh push_task/config/train_config.json 
```
Training will early stop once the student reaches 90 percent success rate in the given task. 
Trained model will be saved in specified folder in `train_confg.json`
## Evaluation :chart_with_upwards_trend:
```
./test_push_critiq.sh push_task/config/test_config.json
```
User can specify the model to test with in `test_config.json`, and video of the task scene will automatically recorded with success/fail label in each trial.  

### Citation :raised_hands:
If you build on our work or find it useful, please cite it using the following bibtex.

```bibtex
@article{kim2025distilling,
  title={Distilling Realizable Students from Unrealizable Teachers},
  author={Kim, Yujin and Chin, Nathaniel and Vasudev, Arnav and Choudhury, Sanjiban},
  journal={arXiv preprint arXiv:2505.09546},
  year={2025}
}
```