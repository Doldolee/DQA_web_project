from flask import Flask, render_template, request, session, redirect
from datetime import date, timedelta
from .import daum_reviews
from .import naver_reviews
from .dictionary import Dictionary
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from openpyxl import load_workbook
from itertools import chain
import joblib
import pandas as pd
import csv
import sys
import io
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
import xlrd

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

#from app import app

from .import dataprocess
from .import decision_model

class Crawler:
    def crawl(self):
        url_3= request.form['url_input']
        
        #url_3="https://news.naver.com/main/read.nhn?mode=LSD&sid1=001&oid=023&aid=0003547854"
        print(url_3)
        if "naver" in url_3:
            naver_reviews.capture_replys_naver(url_3, excel_name="raw_review")
        else:
            daum_reviews.capture_replys_daum(url_3, excel_name="raw_review")

        # 엑셀 리스트로 읽어들이기
        # wb = load_workbook("C:/source_code/bigtering_202.10.10/excelfile/raw_review.csv", read_only=True)
        # data1 = wb.active
        script = []
        # for row in data1:
        #     for cell in row:
        #         var1 = cell.value
        #         result = isinstance(var1, int) 
        #         if result:
        #           break
        #         else: 
        #           script.append(var1)

        f = open('C:/source_code/tensorflow/bigtering_202.10.10/excelfile/raw_review.csv', 'r', -1, 'utf-8')
        rdr = csv.reader(f)
        for line in rdr:
          for x in line:
            script.append(x)
        f.close()


        #(추가) 한글사전 추가
        '''excel_data_df = pd.read_excel('korean_dict.xlsx', sheet_name='NIADic')
        korean_dict_list = excel_data_df['term'].tolist()
        dictionary = Dictionary(korean_dict_list)
        print('[한글 사전 준비 완료]')'''


        #(추가) 전처리. 불용어 거르고 명사화
        #noun_list = []
        #noun_list = dictionary.preprocess_string(script)

        dictionary = Dictionary()


        
        # 전체 오류단어 리스트 생성

        #학습시키기 위한 데이터 전처리 
        dataprocess.data_process(script)

        #모델 학습 decision tree
        error_count, normal_count, score = decision_model.decision_tree()
        print( error_count, normal_count, score)

        #모델 사용      ........안불러와짐...
        # x = pd.read_csv("C:/source_code/tensorflow/bigtering_202.10.10/excelfile/var_x.csv", encoding='utf-8', index_col='단어')
        # clf_from_joblib = joblib.load('function/decisionmodel.pkl') 

        # model_result = list(clf_from_joblib.predict(x))
        # raw_review = list(script)



        ##직접 지정한 오류 사전을 돌려서 나온 결과값
        errorfre_list = []
        for one_line in script:
            i = dictionary.find_dict(one_line)[1]
            errorfre_list.append(i)
        errorfre_list = list(chain.from_iterable(errorfre_list))

        # 오류단어 단어 dict형태로 분류
        pred = Counter(errorfre_list).most_common()
        err_dict = dict(pred)
        data_key=list(err_dict.keys())
        data_value=list(err_dict.values())



        #도넛 차트 변수
        ra_1 = len(errorfre_list)
        cnt_1 = 0
        for i in script:
            cnt_1 +=1
        ratio= [cnt_1/(cnt_1 + ra_1), ra_1/(ra_1 + cnt_1)]



        return render_template("generic.html" ,ratio=ratio, data=data_key, data1=data_value)
