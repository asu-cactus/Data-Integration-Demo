
import pandas as pd
import re
import random

from nltk.tokenize import word_tokenize
from utils.data_utils import clean_str
from utils.Google_KG_Search_API import gen_related_word_corpus


def gen_training_dataset(dataset1_path,dataset2_path):

    time_ser = pd.read_csv(dataset1_path, sep=',' ,header=0 ,dtype={'Province/State': str ,'Country/Region' :str}
                           ,low_memory=False)
    google_mob = pd.read_csv(dataset2_path ,sep=',' ,header=0
                             ,dtype={'country_region': str ,'sub_region_1' :str ,'date' :str}, low_memory=False)

    sub_reg_time_ser = []
    country_time_ser = []
    date_time_ser = []
    label_for_region = []

    value_time_ser = []
    value_time_ser_list = []
    value_gmr = []
    value_gmr_list = []


    for col in time_ser:
        date_time_ser.append(col)
    date_time_ser = date_time_ser[4:]


    label = 0

    for sta, coun in zip(time_ser['Province/State'] ,time_ser['Country/Region']):

        sub_reg_time_ser.append(sta)
        country_time_ser.append(coun)
        label += 1
        label_for_region.append(label)

    # for value content in time series dataset
    for j in date_time_ser:
        col = list(time_ser[j])

        for i in range(0, len(sub_reg_time_ser)):
            if col[i] != 0:
                if pd.isnull(sub_reg_time_ser[i]):      # no sub region

                    value_time_ser.append('country_region,' +
                                   str(country_time_ser[i]) + ',state_subregion, ,date,' + 'confirmed, ')
                    value_time_ser_list.append([country_time_ser[i] ,' ' ,j ,col[i]])
                else:
                    value_time_ser.append('country_region, '+
                                   str(country_time_ser[i] ) +',state_subregion, '+
                                   str(sub_reg_time_ser[i] ) +',date, ' +'confirmed, ')
                    value_time_ser_list.append([country_time_ser[i], sub_reg_time_ser[i], j, col[i]])

    # for value content in Google mobility report dataset
    for coun2, sta2, sub, dat, val in zip(google_mob['country_region'],
                                         google_mob['sub_region_1'],
                                         google_mob['sub_region_2'],
                                         google_mob['date'],
                                         google_mob['transit_stations_percent_change_from_baseline']):

        if not pd.isnull(val):
            if coun2 == 'Canada' or coun2 == 'Australia' or coun2 == 'United States':  ## these countries have sub region
                if pd.isnull(sub):
                    if pd.isnull(sta2):

                        value_gmr.append(
                            'country_region,' + str(coun2) + ',state_subregion, ,date,' + 'transit, ' )

                        value_gmr_list.append([coun2 ,' ' ,dat, val])
                    else:
                        value_gmr.append('country_region,' +
                                         str(coun2) + ',state_subregion,' + str(sta2) + ',date,' + ',transit, ')
                        value_gmr_list.append([coun2, sta2, dat, val])

            else:
                if pd.isnull(sta2):
                    value_gmr.append('country_region,' + str(coun2) + ',state_subregion, ,date,' + ',transit, ')
                    value_gmr_list.append([coun2, ' ', dat, val])

    region_label_time_ser = {}  # region: total length = 310
    region_label_gmr = {}  # region: total length = 204
    attr_label = [1, 2]  # attribute name: confirmed case,transit percentage
    temp_region = []

    for i, j in zip(country_time_ser[:310], sub_reg_time_ser[:310]):
        if pd.isnull(j):
            temp_region.append((i, ' '))
        else:
            temp_region.append((i, j))

    for i in range(len(temp_region)):
        region_label_time_ser[temp_region[i]] = 3 + i

    count = 1
    for row2 in value_gmr_list:
        if row2[0] == 'United States':
            row2[0] = 'US'
        if not region_label_gmr.__contains__((row2[0], row2[1])):
            if (row2[0], row2[1]) in temp_region:
                label = region_label_time_ser[(row2[0], row2[1])]
            else:
                label = 312 + count
                count += 1
                temp_region.append((row2[0], row2[1]))

            region_label_gmr[(row2[0], row2[1])] = label

    # write data into dataset file
    data = []

    for i in range(len(value_time_ser_list)):
        data.append({'value': value_time_ser[i], 'label1': region_label_time_ser[(value_time_ser_list[i][0],
                                                                                  value_time_ser_list[i][1])],
                     'label2': attr_label[0]})

    for i in range(len(value_gmr_list)):
        data.append({'value': value_gmr[i], 'label1': region_label_gmr[(value_gmr_list[i][0], value_gmr_list[i][1])],
                     'label2': attr_label[1]})

    df = pd.DataFrame(data, columns=['value', 'label1', 'label2'])
    dataset_path = '../dataset/training_without_date.csv'

    df.to_csv(dataset_path, index=False)

    return dataset_path


def gen_test_data(test_file_path, related_corpus):
    test_data = pd.read_csv(test_file_path, sep=',', header=0, low_memory=False)

    new_data = []

    for value, label1, label2 in zip(test_data['value'], test_data['label1'],
                                     test_data['label2']):
        new_test_row = ''
        row = value.split(',')
        for words in row:
            if row.index(words) != 5 and row.index(words) != 1 and row.index(words) != 3:
                for word in word_tokenize(clean_str(words)):
                    relate = related_corpus[word]
                    new_test_word = str(random.sample(relate, 1))
                    new_test_row += new_test_word + (',')

            else:

                new_test_row += words + (',')

        new_test_row = clean_str(new_test_row)
        new_test_row = new_test_row.replace('[', ' ')
        new_test_row = new_test_row.replace(']', ' ')

        new_data.append({'value': new_test_row, 'label1': label1, 'label2': label2, })

    df = pd.DataFrame(new_data, columns=['value', 'label1', 'label2'])
    df.to_csv('./dataset/testing_without_date.csv', index=False)


if __name__ == "__main__":
    dataset1_path = '../dataset/source_dataset/time_series_19-covid-Confirmed.csv'

    dataset2_path = "../dataset/source_dataset/Global_Mobility_Report.csv"

    dataset_path = gen_training_dataset(dataset1_path,dataset2_path)

    data = pd.read_csv(dataset_path, sep=',', header=0, low_memory=False)
    related_corpus = gen_related_word_corpus(data)
    gen_training_dataset(dataset_path, related_corpus)