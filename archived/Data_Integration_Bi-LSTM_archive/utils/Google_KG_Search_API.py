from __future__ import print_function
import random
import json
import urllib.parse
import urllib.request
import fasttext
import pandas as pd



from utils.data_utils import clean_str
from nltk.tokenize import word_tokenize







def gen_date_corpus():
    '''
    generate similar date format corpus

    :return: date (type: list)
    '''
    date = []

    all_month = ['January','February','March', 'April','May','June',
                 'July','August','September', 'October', 'November', 'December']


    for month in range(1,13):
        if month == 2:
            day_list = [i for i in range(1,30)]
        elif month == 4 or month==6 or month==9 or month==11:
            day_list = [i for i in range(1,31)]
        else:
            day_list = [i for i in range(1, 32)]

        for day in day_list:

            if day <10 and month <10:
                one_day = ['2020-0{}-0{}'.format(month, day), '{}/{}/2020'.format(month, day),
                           '2020-{}-{}'.format(month, day),
                           '0{}/0{}/2020'.format(month, day), '{}/{}/20'.format(month, day),
                           '{} {}th 2020'.format(all_month[month - 1], day)]
            elif day <10 and month >9:
                one_day = ['2020-{}-0{}'.format(month,day),'{}/0{}/2020'.format(month,day),
                           '2020-{}-0{}'.format(month,day),
                           '{}/0{}/2020'.format(month,day),'{}/0{}/20'.format(month,day),
                           '{} {}th 2020'.format(all_month[month-1],day)]
            elif day>9 and month<10:
                one_day = ['2020-0{}-{}'.format(month, day), '0{}/{}/2020'.format(month, day),
                           '2020-0{}-{}'.format(month, day),
                           '0{}/{}/2020'.format(month, day), '{}/{}/20'.format(month, day),
                           '{} {}th 2020'.format(all_month[month - 1], day)]
            else:
                one_day = ['2020-{}-{}'.format(month, day), '{}/{}/2020'.format(month, day),
                           '2020-{}-{}'.format(month, day),
                           '{}/{}/2020'.format(month, day), '{}/{}/20'.format(month, day),
                           '{} {}th 2020'.format(all_month[month - 1], day)]
            date.append(one_day)
    return date



def gen_related_word_corpus(data):
    '''
    using Google KG API to query similar word

    :param data: original dataset
    :return: related_corpus(type:dictionary)
    '''



    original_corpus = data['value']

    related_corpus = {}
    all_word = []
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

    for item in original_corpus:
        row = item.split(',')

        for words in row:
            if row.index(words) != 5:
                for word in word_tokenize(clean_str(words)):
                    if word not in all_word:
                        all_word.append(word)
                        related_word = []
                        related_word.append(word)

                        # Google Knowledge Graph Search API
                        query = word
                        params = {
                            'query': query,
                            'limit': 2,
                            'indent': True,
                            'key': 'AIzaSyBjLdpfLQyG8yKPTCtj3WMDrNEJ8clakSU',
                        }
                        url = service_url + '?' + urllib.parse.urlencode(params)
                        response = json.loads(urllib.request.urlopen(url).read())
                        for element in response['itemListElement']:

                            if element['resultScore'] > 100: # remove the less relevant words
                                related_word.append(element['result']['name']) # add related word return by KG

                        related_corpus[word]= related_word

    print(related_corpus)
    with open("dataset/dict/related_word_corpus.json","w") as f:
        json.dump(related_corpus,f)

    return related_corpus



def gen_word_embedding_dataset(data):
    '''
    generate the word corpus used for word embedding training

    :param data: original dataset
    :return: word corpus path
    '''

    with open("dataset/dict/related_corpus.json","r") as f:
        related_corpus = json.load(f)


    # write word embedding training dataset
    embedding_corpus_path = 'dataset/new_corpus_without_date'
    f=open(embedding_corpus_path, "w",encoding='utf-8')

    # data = pd.read_csv(dataset_path,sep=',',header=0, low_memory=False)
    corpus = data['value']
    for item in corpus:
        for n in range(0,10):
            new_row = ''
            row = item.split(',')
            for words in row:
                if row.index(words) != 5 and row.index(words) != 1 and row.index(words) != 3:
                    for word in word_tokenize(clean_str(words)):
                        relate = related_corpus[word]
                        new_word = str(random.sample(relate,1))
                        new_row += str(new_word)
                else:

                    new_row += str(words)

            new_row = clean_str(new_row)

            f.write(str(new_row))
            f.write('\n')

    return embedding_corpus_path


def train_embedding_model(embedding_corpus_path):
    '''
    train word embedding model using FastText

    :param embedding_model_path:
    :return: word embedding model (type: binary file)
    '''
    model = fasttext.train_unsupervised(embedding_corpus_path, 'skipgram',
                                        epoch=10, minn=2, maxn=5, dim=150, thread=16)
    model.save_model('pretrained_embedding_model/new_corpus_without_date.bin')


if __name__ == "__main__":
    dataset_path = "dataset/training_without_date.csv"
    data = pd.read_csv(dataset_path,sep=',',header=0, low_memory=False)


    date_corpus = gen_date_corpus()
    related_word_corpus = gen_related_word_corpus(data)
    embedding_model_path = gen_word_embedding_dataset(data)

    print('training word embedding model')
    train_embedding_model(embedding_model_path)