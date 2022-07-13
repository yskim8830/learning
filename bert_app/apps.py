from django.apps import AppConfig
import torch
import configparser
import os
import logging
import gluonnlp as nlp
import pandas as pd
from .util import bert_util

from kobert.utils import get_tokenizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger('my')
global bert_data
class BertAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bert_app'
    def ready(self):
        global bert_data
        if not os.environ.get('APP'):
            os.environ['APP'] = 'True'
            bert_data = get_model()

def get_model():
    properties = configparser.ConfigParser()
    properties.read('prochat.ini')
    config = properties["CONFIG"] ## 섹션 선택
    prochat_path = properties["CONFIG"]["prochat_path"]

    #bert init
    device, bertmodel, vocab = bert_util.set_init()

    #read learning file
    file_name = '@prochat_dialog_question'
    file_ext = r".csv"
    file_list = [_ for _ in os.listdir(prochat_path) if _.endswith(file_ext) & _.startswith(file_name)]
    #print('pid : ', os.getpid(), ' : ', file_list)
    berts_data = {}
    for question_file in file_list:
        site_no = question_file.replace(file_name+'_', '').replace(file_ext, '')
        dataset_cate = pd.read_csv(os.path.join(prochat_path,question_file))
        
        # 라벨링
        encoder = LabelEncoder()
        encoder.fit(dataset_cate['intent'])
        dataset_cate['intent'] = encoder.transform(dataset_cate['intent'])
        
        # 라벨링된 카테고리 매핑
        mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
        mapping_len = len(mapping)
        
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        
        modelload = bert_util.BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=mapping_len).to(device)
        modelload.load_state_dict(torch.load(os.path.join(prochat_path,'learningModel_'+site_no+'.pt'), device))
        
        data_set = {'modelload' : modelload, 'mapping' : mapping, 'tok' : tok, 'device' : device}
        logger.info('bert data set, site_no is ' + str(site_no) + ' : load success.')
        
        
        berts_data[site_no] = data_set
    return berts_data