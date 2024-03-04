import imghdr
import os
from pathlib import Path
from pydoc import describe
import random
import re
import jieba
from numpy import NAN
import pandas as pd
import requests
import jieba.posseg as pseg
import sys
import warnings
warnings.filterwarnings("ignore")

from models import *

def reform_float_to_str(float_num):
    str_num = str(float_num)
    str_num = str_num.replace('.0', '')
    return str_num


def reform_description(description):
    description = description.strip()
    description = description.replace('\n', '')
    return description


def reform_img_url(img_url):
    img_url = img_url.replace(' ', '')
    img_url = img_url.replace(';', '\n')
    img_url = img_url.replace(',', '\n')
    img_url = img_url.split('\n')
    return img_url



def extract_keyword(report):
    f_stopwords = open('stopwords1893.txt', encoding='utf-8')
    stopwords = f_stopwords.read()
    stopwords = stopwords.split('\n')

    report.keyword = jieba.lcut(report.description, cut_all=False)
    keyword_1 = []
    for word in report.keyword:  # filter stopwords
        if word not in stopwords and word != ' ':
            keyword_1.append(word)
    report.keyword = keyword_1


def load_caselist(project):
    csvPath = 'data/reports/' + project + '.csv'    
    df=pd.read_csv(csvPath, encoding='utf-8')
    df = df.fillna(1)
    nrows = df.shape[0]

    caselist = []

    report_id_col_index = 0
    bug_id_col_index = 1
    bug_category_col_index = 2
    severity_col_index = 3
    recurrent_col_index = 4
    bug_page_index = 6
    report_title_index = 7
    description_col_index = 8
    img_url_col_index = 9
    test_case_name_index = 10
    test_case_front_index = 11
    test_case_behind_index = 12
    test_case_description_index = 13
    
    case_id = 0
    for row in range(0, nrows):
        newcase = case(case_id)
        newcase.description = reform_description(df.iloc[[row]].values[0][description_col_index])
        newcase.shotlist = df.iloc[[row]].values[0][img_url_col_index]
        newcase.severity = reform_float_to_str(df.iloc[[row]].values[0][severity_col_index])
        newcase.recurrent = reform_float_to_str(df.iloc[[row]].values[0][recurrent_col_index])
        newcase.bug_page = df.iloc[[row]].values[0][bug_page_index]
        newcase.report_title = reform_description(df.iloc[[row]].values[0][report_title_index])
        newcase.test_case_name = reform_description(df.iloc[[row]].values[0][test_case_name_index])
        newcase.test_case_front =  reform_description(df.iloc[[row]].values[0][test_case_front_index])
        newcase.test_case_behind =  reform_description(df.iloc[[row]].values[0][test_case_behind_index])
        newcase.test_case_description =  reform_description(df.iloc[[row]].values[0][test_case_description_index])

        if newcase.shotlist!=1 and  newcase.shotlist.strip() != '':
             newcase.shotlist = reform_img_url(newcase.shotlist)
        caselist.append(newcase)
        case_id += 1
    for report in caselist:
        extract_keyword(report)
    return caselist


def statistics_case(caselist,proName):
    # |F|,|S|,|Rs|

    bug_severity = {}
    bug_severity[1] = 0
    bug_severity[2] = 0
    bug_severity[3] = 0
    bug_severity[4] = 0
    bug_severity[5] = 0

    
    report_num = 0
    for case in caselist:
        bug_severity[int(case.severity)] += 1
        report_num += 1
    
    
    for key in bug_severity.keys():
        print(key,end="\t")
        print(bug_severity[key])
        
    random.shuffle(caselist)
    train_rates = ["0.8"]
    for rate in train_rates:
        limit_dict = {1:0, 2:0, 3:0, 4:0, 5:0}
        labels_list = []
        des_list = []
        my_method_list = []
        train_or_test_list = []
        for case in caselist:
            des_list.append(case.description)
            labels_list.append(case.severity)
            my_method_list.append(case.description + case.recurrent + case.bug_page )

            limit_dict[int(case.severity)] += 1
            if limit_dict[int(case.severity)] <= int(bug_severity[int(case.severity)] * float(rate)):
                train_or_test_list.append('train')
            else:
                train_or_test_list.append('test')

        dataset_name_1 = proName + '_' + 'my_method'+ "_" + rate
        meta_data_list = []
        for i in range(len(my_method_list)):
            meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels_list[i]
            meta_data_list.append(meta)
        meta_data_str = '\n'.join(meta_data_list)
        f1 = open('data/' + dataset_name_1 + '.txt', 'w',encoding='utf-8')
        f1.write(meta_data_str)
        f1.close()
        corpus_str = '\n'.join(my_method_list)
        f2 = open('data/corpus/' + dataset_name_1 + '.txt', 'w',encoding='utf-8')
        f2.write(corpus_str)
        f2.close()

    n = len(caselist)
    return n

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Use: python build_graph.py <dataset>")
    pro = sys.argv[1]
    caselist = load_caselist(pro)
    n = statistics_case(caselist,pro)


