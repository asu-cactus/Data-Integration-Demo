# Data Integration Bi-LSTM Model

Table cell value prediction using Bi-LSTM model 





## Environment Requirement

python 3.6+

TensorFlow 1.15

sklearn

nltk

FastText



## Dataset

#### Covid-19 scenario

- Covid-19 dataset  https://github.com/CSSEGISandData/COVID-19

- Google Mobility Report dataset  https://www.google.com/covid19/mobility/



#### Machine log scenario 

- Linux log data
- macOS log data
- Android log data



## Model Architecture

### Pretrained Word Embedding + Bi-LSTM

Pretrained Embedding Layer (based on fastText) 

| Word Embedding Size | Subword minimum size | Subword maximum size |
| ------------------- | -------------------- | -------------------- |
| 150                 | 2                    | 5                    |



Bi-LSTM

- optimizer: Adam optimizer
- loss function: Cross Entropy Loss
- dropout function


| Input Size | Bi-LSTM Layer Size(forward & Back) | FC Layer size |
| ---------- | ---------------------------------- | ------------- |
| 150        | 512                                | 256           |



Training hyperparameters

| Learning rate | Batch Size | Dropout probability |
| ------------- | ---------- | ------------------- |
| 0.001         | 64         | 0.8                 |



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











 



