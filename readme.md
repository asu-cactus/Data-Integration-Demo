# Data Integration Demo 

### Data Integration for Data with Schema Changes 

1.  Attribute Renaming
2.  Cell Value Reformatting
3.  Key Expansion
4.  Key Order Changing  



### Table Cell Position Prediction Using Sequence/Language Model

1.  Column (attribute) prediction 
2.  Key index prediction
3.  Aggregation mode prediction (only apply to covid-19 dataset)



## Dataset

#### Covid-19 scenario

- Covid-19 dataset  https://github.com/CSSEGISandData/COVID-19

- Google Mobility Report dataset  https://www.google.com/covid19/mobility/



#### Machine log scenario 

- Linux log data
- macOS log data
- Android log data




## Model

- [Pretrained Word Embedding (fastText) + Bi-LSTM](https://github.com/asu-cactus/Data-Integration-Demo/blob/master/Bi_LSTM/)
- [Pretrained Word Tokenizer (BERT Tokenizer) + Encoder-only Transfomer](https://github.com/asu-cactus/Data-Integration-Demo/blob/master/Transformer/)



## Usage

Dataset:

[Github Repository](https://github.com/asu-cactus/Data-Integration-Demo/blob/master/Data_Integration_Dataset/)

[Google Drive](https://drive.google.com/drive/folders/19oLAKktjI0uk8v4lcdBTnRBTyqN-tGeR?usp=sharing)



Download dataset: 

```python
import gdown
gdown.download('https://drive.google.com/drive/folders/19oLAKktjI0uk8v4lcdBTnRBTyqN-tGeR', output=None, quiet=False)
```



Run with [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index) or local Jupyter Notebook.









 



