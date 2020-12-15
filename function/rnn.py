import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import os
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


train_data = pd.read_csv('train_rawdata.csv',  header=0 )
test_data = pd.read_csv('test_rawdata.csv', header=0)

##중복된 데이터 확인##
# print(train_data['document'].nunique(), train_data['label'].nunique())

##중복 샘플 제거##
train_data.drop_duplicates(subset=['document'], inplace=True)

## 레이블 값 분포 확인##
# print(train_data.groupby('label').size().reset_index(name = 'count'))

##null 값 확인 ##
# print(train_data.isnull().values.any())

##한글과 공백 제외하고 모두 제거##
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# print(train_data[:5])

##빈 값 행을 null값으로 변경하고 null존재여부 확인(null값은 영어 댓글) 및 null값 제거##
train_data['document'].replace('', np.nan, inplace=True)
# print(train_data.isnull().sum())
# print(train_data.loc[train_data.document.isnull()][:5])
train_data = train_data.dropna(how = 'any')


##test데이터 전처리##
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
# print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

##토큰화##
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
X_train = []
okt = Okt()
for sentence in train_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

X_test = []
for sentence in test_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

##정수 인코딩##
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# print(tokenizer.word_index)

##데이터에서 희귀 단어 비중 확인##
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

##패딩##
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
#모델이 처리할 수 있도록 X_train과 X_test의 모든 샘플의 길이를 특정 길이로 동일하게 맞춰줄 필요가 있습니다. 
# def below_threshold_len(max_len, nested_list):
#   cnt = 0
#   for s in nested_list:
#     if(len(s) <= max_len):
#         cnt = cnt + 1
#   print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 30
# below_threshold_len(max_len, X_train)
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

##LSTM으로 리뷰 오류 분석##

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es,mc], batch_size=60, validation_split=0.2)


# loaded_model = load_model('best_model.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# def sentiment_predict(new_sentence):
#   new_sentence = okt.morphs(new_sentence) # 토큰화
#   new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
#   encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
#   pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
#   score = float(loaded_model.predict(pad_new)) # 예측
#   if(score > 0.5):
#     print("{:.2f}% 확률로 오류 리뷰입니다.\n".format(score * 100))
#   else:
#     print("{:.2f}% 확률로 정상 리뷰입니다.\n".format((1 - score) * 100))

