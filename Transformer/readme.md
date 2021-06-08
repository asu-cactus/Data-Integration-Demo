# Data Integration Transformer Model

Table cell value prediction using Vanilla Transformer model 





## Environment Requirement

python 3.6+

PyTorch 

HuggingFace Transfomer

Sklearn



## Dataset

#### Covid-19 scenario

- Covid-19 dataset  https://github.com/CSSEGISandData/COVID-19

- Google Mobility Report dataset  https://www.google.com/covid19/mobility/



#### Machine log scenario 

- Linux log data
- macOS log data
- Android log data



## Model Architecture

### Pretrained Word Tokenizer + Transformer

Word Tokenizer

- Pretrained BERT base Tokenizer



Encoder-only Vanilla Transformer

- optimizer: Adam optimizer
- loss function: Cross Entropy Loss
- dropout function


| Num of Layers (Blocks) | Num of Attention heads | Block Size |
| ---------------------- | ---------------------- | ---------- |
| 12                     | 8                      | 128        |



Training hyperparameters

| Learning rate | Batch Size | Dropout probability |
| ------------- | ---------- | ------------------- |
| 0.0005        | 128        | 0.9                 |



## Usage

Dataset:

[Github Repository](https://github.com/asu-cactus/Data-Integration-Demo/blob/master/Data_Integration_Dataset/)

[Google Drive](https://drive.google.com/drive/folders/19oLAKktjI0uk8v4lcdBTnRBTyqN-tGeR?usp=sharing)



Download dataset: 

```python
import gdown
gdown.download('https://drive.google.com/drive/folders/19oLAKktjI0uk8v4lcdBTnRBTyqN-tGeR', output=None, quiet=False)
```



Run with Google Colab or local Jupyter Notebook.









 



