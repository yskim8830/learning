import json
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .util.bert_util import bertQuestion
from .learn import learning

from django.http import Http404

class trainToDev(APIView):
    def get(self , request):
        dicPath = request.query_params.get('dicPath') 
        print(dicPath)
        esUrl = request.query_params.get('esUrl','127.0.0.1:6251')
        print(esUrl)
        debug = request.query_params.get('debug',False)
        
        result_dic = learning.learningBERT(dicPath,esUrl,debug)
        if result_dic.result == 'fail':
            Response(result_dic)
            
        return Response(result_dic)
    
    def post(self , request):
        data = json.loads(request.body) #파라미터 로드
        mode = str(data['mode'])
        result_dic = {} #결과 set
        if mode == 'run' or mode == 'train':
            result_dic = learning.learningBERT(data)
            if result_dic['result'] == 'fail':
                return Response(result_dic)
        elif mode == 'send' or mode == 'dist':
            result_dic = {'준비중'}
        elif mode == 'clear':
            result_dic = {'준비중'}
        elif mode == 'runstop':
            result_dic = {'준비중'}
        elif mode == 'dic':
            result_dic = learning.save_dict(data)
            
        return Response(result_dic)
    
class bert_Question(APIView):
    def get(self , request):
        site_no = request.query_params.get('siteNo') 
        print(site_no)
        question = request.query_params.get('query')
        print(question)
        bert_query = bertQuestion(site_no)
        result_answer = bert_query.question(question)
    
        return Response(result_answer)