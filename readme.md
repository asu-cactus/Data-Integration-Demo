# Data Integration Demo 

### Data Integration for Fast-Evolving Data Sources 

1. Dimension Pivoting Changes
2. Attribute Name Changes
3. Attribute Value Changes

## Environment Requirement

python 3.6+

TensorFlow 1.14

nltk

FastText



## Dataset

###### two tables has similar but not same table schema, also with frequent schema evolving.



time_series_19-covid-Confirmed.csv  https://github.com/CSSEGISandData/COVID-19

Global_Mobility_Report.csv  https://www.google.com/covid19/mobility/



| Number of Rows | Number of Columns(Attributes)                                |
| -------------- | ------------------------------------------------------------ |
| 28751          | 5                                                                                                                           (country_region, state_subregion, date, confirmed_case, transit_percentage) |





## Model Architecture

### Word RNN

Pretrained Embedding Layer + bi-LSTM + FC

| Embedding Layer | bi-LSTM | Fully Connected Layer |
| --------------- | ------- | --------------------- |
| Vocabulary Size | 512     | 256                   |



##### Word Embedding Using FastText

Pretrained Corpus using Wikipedia English content: vocabulary size = 2519370

Customized Corpus using data table based words:   vocabulary size = 535



## Usage

### 

Three models are separated into three notebook files 

you can run the notebook file on Google Colab 











 



