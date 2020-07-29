import fasttext
import pandas as pd
import numpy as np
import re

from nltk.tokenize import word_tokenize


def clean_str(text):
    '''
    regular expression to clean text file

    '''
    text = re.sub(r"[_,.!?’\'\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    text = text.lower()

    return text

def create_onehot_labels(labels_index,num_labels):
    '''
    create onehot label vector

    :param labels_index: preset order of label
    :param num_labels: number of classes
    :return: onehot label vector
    '''
    label = [0] * num_labels

    for item in labels_index:

        label[int(item-1)] = 1
    return label


def cos_sim(vector_a, vector_b):
    """
    calculate the cosine similarity between two vectors

    :param vector_a: vector a
    :param vector_b: vector b
    :return: cosine similary
    """

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim




def train_data_word2vec(TRAIN_PATH,num_class,vocab_size, embed_size, embedding_model):
    '''
    create the training set(train_x) and labels(train_y)

    :param TRAIN_PATH: training data file
    :param num_class: total number of classes(rows + attribute)
    :param vocab_size: number of total vacabulary in pretrained embedding
    :param embed_size: embedding size for each word(word vector dimension)
    :param embedding_model: pretrained embedding model file
    :return: content_index_list, word vector index in embedding matrix
             onehot_labels_list, word label vector
             trainset_embedding_matrix, oov word(from training dataset) embedding matrix
             oov_word, oov word in training dataset
    '''

    model = fasttext.load_model(embedding_model)
    vocab = dict([(word, model.get_word_id(word)) for word in model.get_words()])


    df = pd.read_csv(TRAIN_PATH, names=["content", "label1", "label2"], sep=',', header=0)


    content_index_list = []
    onehot_labels_list = []

    trainset_embedding_matrix = np.zeros((0,embed_size))
    oov_word = []

    count = 0
    for row in range(len(df['content'])):

        content = df['content'][row]

        result = []

        for item in word_tokenize(clean_str(content)):
            word2id = vocab.get(item)
            if word2id is None and item not in oov_word:
                oov_word.append(item)

                word_vec = model.get_word_vector(item)
                trainset_embedding_matrix = np.insert(trainset_embedding_matrix,
                                                      len(trainset_embedding_matrix), values=word_vec, axis=0)
                word2id = len(model.get_words()) +count
                count +=1

            elif word2id is None and item in oov_word:
                word2id = vocab_size + oov_word.index(item)

            result.append(word2id)
        content_index_list.append(result)


    print('flag',count)
    label1 = df["label1"]
    label2 = df['label2']
    # label3 = df['label3']


    for l1, l2 in zip(label1, label2):
        label = [l1, l2]

        onehot_labels_list.append(create_onehot_labels(label, num_class))


    print('length of trainset_matr',len(trainset_embedding_matrix))
    print('oov word',oov_word)
    print('x',content_index_list[:10])

    with open('../dataset/embedding_matrix/trainset_embedding_matrix.npy', 'wb') as f:
        np.save(f, trainset_embedding_matrix)
    return content_index_list, onehot_labels_list, trainset_embedding_matrix,oov_word


def test_data_word2vec(TEST_PATH,num_class,vocab_size,embedding_model,oov_word):
    '''
    create the testing set(test_x) and labels(test_y)

    :param TEST_PATH: test dataset
    :param num_class: total number of classes(rows + attribute)
    :param vocab_size: number of total vocabulary in pretrained embedding
    :param embedding_model: pretrained embedding model file
    :param oov_word: oov word in training dataset

    :return: content_index_list, word vector index in embedding matrix
             onehot_labels_list, word label vector

    '''


    model = fasttext.load_model(embedding_model)
    vocab = dict([(word, model.get_word_id(word)) for word in model.get_words()])

    oov_vocab = dict([(word, oov_word.index(word)+vocab_size) for word in oov_word])
    whole_vocab ={}
    whole_vocab.update(vocab)
    whole_vocab.update(oov_vocab)
    print('len of whole voc',len(whole_vocab))

    df = pd.read_csv(TEST_PATH, names=[ "content", "label1", "label2"], sep=',', header=0)


    content_index_list = []
    onehot_labels_list = []

    oov_list = []


    for row in range(len(df['content'])):

        content = df['content'][row]

        result = []

        for item in word_tokenize(clean_str(content)):
            word2id = whole_vocab.get(item)
            if word2id is None:

                word2id = 0
                oov_list.append(item)
            result.append(word2id)
        content_index_list.append(result)


    label1 = df["label1"]
    label2 = df['label2']


    for l1, l2 in zip(label1, label2):
        label = [l1, l2]

        onehot_labels_list.append(create_onehot_labels(label, num_class))

    print('length of oov words in test set',len(oov_list))
    print('oov word',oov_list)

    print('test_x', content_index_list[:10])



    return content_index_list, onehot_labels_list


def load_word2vec_matrix(embedding_model):
    '''
    create pretrained word embedding matrix

    :param embedding_model: pretrained embedding model file

    :return:
             vocab_size, number of total vacabulary in pretrained embedding
             embedding_size, embedding size for each word(word vector dimension)
             embedding_matrix, word embedding matrix
    '''


    model = fasttext.load_model(embedding_model)
    vocab_size = (model.get_output_matrix()).shape[0]
    embedding_size = model.get_dimension
    print('vocabulary size',vocab_size)



    vocab = dict([(word, model.get_word_id(word)) for word in model.get_words()])

    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for word, index in vocab.items():
        if word is not None:
            embedding_matrix[index] = model[word]
    with open('../dataset/embedding_matrix/embedding_matrix.npy', 'wb') as f:
        np.save(f, embedding_matrix)
    return vocab_size, embedding_size, embedding_matrix



def batch_iter(inputs, outputs, batch_size, num_epochs):
    '''

    :param inputs: unbatched data
    :param outputs: batched data
    :param batch_size: size of every data batch
    :param num_epochs: number of epochs

    :return:
           A batch iterator for data set
    '''
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]




if __name__ == "__main__":


    # #word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True)
    # word2vec_model = word2vec.Word2Vec.load('dataset/word2vec_100.model')
    # content_index_list, onehot_labels_list = data_word2vec('train', 460, word2vec_model)
    # pad_seq = pad_sequences(content_index_list,maxlen=10, value=0.)
    # print(len(content_index_list))
    # print(content_index_list[:10])
    # print(pad_seq[:10])
    # vocab_size, embedding_size, embedding_matrix = load_word2vec_matrix(MODEL_PATH)
    # print(vocab_size,embedding_size)
    # print(embedding_matrix.shape)



    #
    # model = fasttext.load_model("dataset/wiki.en.bin")
    #
    # print(model.get_subwords('2020'))
    # print(model.get_word_vector('20'))
    # print(model.get_word_vector('20>'))
    # print('cos',cos_sim(model.get_word_vector('country'),model.get_sentence_vector('nationality')))
    #
    # fr = open('dataset/new_corpus_without_date&add_change', "r")
    # fw = open('dataset/new_corpus_without_date&add_change_v2', "w",encoding='utf-8')
    #
    #
    # for line in fr.readlines():
    #     new_line = re.sub(r"[,!?\'\"]", " ", line)
    #     new_line = re.sub(r"\s{2,}", " ", new_line)
    #     new_line = new_line.replace('[','')
    #     new_line = new_line.replace(']', '')
    #
    #     new_line = new_line.strip().lower()
    #
    #     fw.write(new_line)
    #     fw.write('\n')

    # model = fasttext.train_unsupervised('dataset/new_corpus_without_date','skipgram',epoch = 10,minn=2, maxn=5, dim=150,thread=16)
    # model.save_model('pretrained_embedding_model/new_corpus_without_date.bin')
    l_model = fasttext.load_model("../pretrained_embedding_model/new_corpus_without_date.bin")
    print(l_model.get_nearest_neighbors('country',10))
    # print(model.get_words('region'))

    # dist = np.sqrt(np.sum(np.square(model.get_word_vector('2020'),model.get_word_vector('20'))))
    # print('eu distance', dist)
    # # print(model.get_nearest_neighbors('list'))
    # # print(model.get_word_vector('the'))
    # print(model.get_word_id('2009'))
    # # print(len(model.get_words()),len(model.get_labels()))
    # print(model.get_word_vector('20'))
    # # word = model.get_word_vector('threshholds')
    # # # print(word)
    # word2vec_model = FastText.load_fasttext_format('dataset/wiki.en')
    # # print(model.get_word_vector('US'))
    # word2vec_model = KeyedVectors.load_word2vec_format('dataset/wiki.en.vec')
    #
    # print('111',word2vec_model.get_nearest_neighbors('confirmed'))
    # #print(model.wv.most_similar('United States'))
    # vocab_size, embedding_size, embedding_matrix = load_word2vec_matrix("dataset/wiki.en.bin")
    # content_index_list, onehot_labels_list, trainset_embedding_matrix =  data_word2vec('train',460,"dataset/wiki.en.bin")
    # print(embedding_matrix.shape,type(embedding_matrix))
    # print(trainset_embedding_matrix.shape)


