import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

import os
import logging
import pandas as pd

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder

import json
logger = logging.getLogger('my')

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#device를 설정하고 KOBERT 를 로드한다.
def set_init():
    global device
    global bertmodel, vocab
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info('device set GPU')
    else:
        device = torch.device('cpu')
        logger.info('device set CPU')
    bertmodel, vocab = get_pytorch_kobert_model()
    
    return device, bertmodel, vocab

class bert:
    def __init__(self, site_no, path, question_file):
        self.site_no = site_no
        self.intent_path = path
        self.question_file = question_file
        
    def learning(self):
        ret = True
        try :
            global device
            global bertmodel, vocab
            self.device = device
            self.bertmodel = bertmodel
            self.vocab = vocab
            
            dataset_train = []
            dataset_test = []
            
            # 학습용 데이터셋 불러오기 
            dataset_train1 = pd.read_csv(os.path.join(self.intent_path,self.question_file))
            logger.info(dataset_train1.head())
            
            #임의의 카테고리를 테스트용으로 셋 (수정 필요)
            data1 = dataset_train1[dataset_train1['intent'] == '인터넷뱅킹_IM뱅크 신청']
            data2 = dataset_train1[dataset_train1['intent'] == '폰뱅킹 신청_해지']
            data3 = dataset_train1[dataset_train1['intent'] == '인터넷뱅킹 본인인증 문자']
            data4 = dataset_train1[dataset_train1['intent'] == '일상대화_자기소개']
            data5 = dataset_train1[dataset_train1['intent'] == '전화번호 조회_기획 담당자 및 연락처']
            new_data = data1.append([data2, data3, data4, data5], sort=False)
            new_data = pd.DataFrame(new_data)
            logger.info(new_data.head())
            
            # 라벨링
            encoder = LabelEncoder()
            encoder.fit(dataset_train1['intent'])
            dataset_train1['intent'] = encoder.transform(dataset_train1['intent'])
            # dataset_train1.head()

            encoder_test = LabelEncoder()
            encoder_test.fit(new_data['intent'])
            new_data['intent'] = encoder_test.transform(new_data['intent'])
            # new_data.head()
            
            # 라벨링된 카테고리 매핑
            mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
            mapping_len = len(mapping)
            logger.info('Mapping intent Length is ' + str(mapping_len))
            #BERT 데이터셋으로 만들기위해 리스트 형으로 형변환
            dataset_train = dataset_train1.values.tolist()
            dataset_test = new_data.values.tolist()
            
            tokenizer = get_tokenizer()
            tok = nlp.data.BERTSPTokenizer(tokenizer, self.vocab, lower=False)
            
            data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
            data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
            
            train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)
            
            model = BERTClassifier(self.bertmodel,  dr_rate=0.5, num_classes=mapping_len).to(self.device)
            
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            
            # 옵티마이저 선언
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            
            t_total = len(train_dataloader) * num_epochs
            warmup_step = int(t_total * warmup_ratio)
            
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
            
            # 모델 학습 시작
            for e in range(num_epochs):
                train_acc = 0.0
                test_acc = 0.0
                
                model.train()
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                    optimizer.zero_grad()
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    valid_length= valid_length
                    label = label.long().to(self.device)
                    out = model(token_ids, valid_length, segment_ids)
                    loss = loss_fn(out, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # gradient clipping
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    train_acc += calc_accuracy(out, label)
                    if batch_id % log_interval == 0:
                        print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                logger.info("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
                
                model.eval() # 평가 모드로 변경
                
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    valid_length= valid_length
                    label = label.long().to(self.device)
                    out = model(token_ids, valid_length, segment_ids)
                    test_acc += calc_accuracy(out, label)
                logger.info("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
            ##모델 학습 끝
            
            #모델 저장 (elasticsearch에저장 가능?)
            torch.save(model.state_dict(), os.path.join(self.intent_path,'learningModel_'+self.site_no+'.pt'))

        except Exception as e:
            ret = False
            logger.error("[trainToDev] BERT error Msg : "+ e)
            logger.error("[trainToDev] BERT error Msg : "+ e.msg)
        finally:
            return ret
        

class bertQuestion:
    def __init__(self, site_no):
        #from ..apps import get_model
        from ..apps import get_model
        bert_data = get_model()
        self.site_no = site_no
        self.data_set = bert_data[site_no]
        
    def question(self, question):
        self.modelload = self.data_set['modelload']
        self.mapping = self.data_set['mapping']
        self.tok = self.data_set['tok']
        self.devices = self.data_set['device']
        
        def getIntent(seq):
            cate = [self.mapping[i] for i in range(0,len(self.mapping))]
            tmp = [seq]
            transform = nlp.data.BERTSentenceTransform(self.tok, max_len, pad=True, pair=False)
            tokenized = transform(tmp)

            self.modelload.eval()
            result = self.modelload(torch.tensor([tokenized[0]]).to(self.devices), [tokenized[1]], torch.tensor(tokenized[2]).to(self.devices))
            idx = result.argmax().cpu().item()
            
            answer = {"question" : seq, "intent" : cate[idx], "reliability" : "{:.2f}%".format(softmax(result,idx))}
            #logger.debug("질의의 카테고리는:", answer["intent"])
            #logger.debug("신뢰도는:", answer["reliability"])
            
            return answer
        
        return getIntent(question)
        
        
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 0, # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.num_classes = num_classes
                 
        self.classifier = nn.Linear(hidden_size , self.num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
# 학습 평가 지표인 accuracy 계산 -> 얼마나 타겟값을 많이 맞추었는가
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100