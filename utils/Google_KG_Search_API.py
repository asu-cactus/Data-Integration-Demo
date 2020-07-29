from __future__ import print_function
import random
import json
import urllib.parse
import urllib.request
import fasttext

from utils.data_utils import clean_str
from nltk.tokenize import word_tokenize




# dataset_path = "dataset/training_without_date.csv"


def gen_date_corpus():

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


    # data = pd.read_csv(dataset_path,sep=',',header=0, low_memory=False)
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

    with open("dataset/dict/related_corpus.json","r") as f:
        related_corpus = json.load(f)


    # write word embedding training dataset
    embedding_model_path = 'dataset/new_corpus_without_date'
    f=open(embedding_model_path, "w",encoding='utf-8')

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

    return embedding_model_path


def train_embedding_model(embedding_model_path):
    model = fasttext.train_unsupervised(embedding_model_path, 'skipgram',
                                        epoch=10, minn=2, maxn=5, dim=150, thread=16)
    model.save_model('pretrained_embedding_model/new_corpus_without_date.bin')

#
# test_file_path  = "dataset/training_without_date.csv"
#
# test_data = pd.read_csv(test_file_path,sep=',',header=0, low_memory=False)
#
# new_data = []
# new_value = []
# for value, label1, label2 in zip(test_data['value'], test_data['label1'],
#                                  test_data['label2']):
#     new_test_row = ''
#     row = value.split(',')
#     for words in row:
#         if row.index(words) != 5 and row.index(words) != 1 and row.index(words) != 3:
#             for word in word_tokenize(clean_str(words)):
#                 # if word in related_corpus:
#
#                 relate = related_corpus[word]
#                 new_test_word = str(random.sample(relate,1))
#                 new_test_row += new_test_word+(',')
#
#         else:
#
#             new_test_row += words+(',')
#
#     new_test_row = re.sub(r"[!_?\'\"]", " ", new_test_row)
#     new_test_row = re.sub(r"!?\'\"]", " ", new_test_row)
#     new_test_row = re.sub(r"\s{2,}", " ", new_test_row)
#     new_test_row = new_test_row.replace('[', ' ')
#     new_test_row = new_test_row.replace(']', ' ')
#
#     new_data.append({'value': new_test_row, 'label1': label1, 'label2': label2,})
#
# df2 = pd.DataFrame(new_data,columns=['value','label1','label2'])
# df2.to_csv('./dataset/testing_without_date.csv',index=False)
#



#
#
#
# query = 'latitude'
# service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
# params = {
#     'query': query,
#     'limit': 3,
#     'indent': True,
#     'key': 'AIzaSyBjLdpfLQyG8yKPTCtj3WMDrNEJ8clakSU',
# }
# url = service_url + '?' + urllib.parse.urlencode(params)
# response = json.loads(urllib.request.urlopen(url).read())
# for element in response['itemListElement']:
#    print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
#

# import http.client, urllib.parse
# import json
#
# subscriptionKey = '9de98e9361c84843a72599eb8f4b9a45'
# host = 'api.cognitive.microsoft.com'
# path = '/bing/v7.0/entities'
# mkt = 'en-US'
# query = 'United States'
#
# params = '?mkt=' + mkt + '&q=' + urllib.parse.quote (query)
#
# headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
# conn = http.client.HTTPSConnection (host)
# conn.request("GET", path + params, None, headers)
# response = conn.getresponse ()
# result = response.read()
# print (json.dumps(json.loads(result), indent=4))