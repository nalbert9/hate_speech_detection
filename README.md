# Hate Speech Detection using Deep Learning 

## Project Overview
The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. In this work we use Transformers to develop tools that could help to improve online conversation. 

## Project Structure
The project is structured as a series of Jupyter notebooks:

__Notebook 0__ : The dataset;

__Notebook 1__ : Training the Model;

__Notebook 2__ : Load trained model and generate predictions.

## Installation

1. Clone the project repo
```sh
$ git clone https://github.com/nalbert9/hate_speech_detection.git
```

2. Install PyTorch 
```sh
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. Install PyTorch Lighting
```sh
$ pip install pytorch-lightning
```
4. Install a few required pip packages
```sh
$ pip install -r requirements.txt
```
## Inference

## References
[BERT](https://arxiv.org/abs/1810.04805), [Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)
