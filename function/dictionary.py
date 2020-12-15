import re
from konlpy.tag import Okt
import pandas as pd
okt = Okt()

class Dictionary:
    noun_list = []
    #korean_dict_list= []
    #def __init__(self,dict):
     #   self.korean_dict_list = dict

    def preprocess_string(self, string_list):
        for line in string_list:
            line = re.sub('\s', " ", line)  #텍스트파일 전처리.
            nouns = okt.nouns(line)
            self.noun_list.append(nouns)
        print('[불용어 처리 및 명사화 완료]')
        return self.noun_list


    # 한 줄의 에러 갯수 카운트
    def find_dict(self, one_line):
        cnt = 0
        errorfre_list = []
        f = open('dict.txt', 'r', encoding='utf-8')
        
        
        while True:
            error_word = f.readline()
            error_word = error_word.strip()
            if not error_word:
                break
            cnt = cnt + one_line.count(error_word)

            # 에러 단어 리스트 반환
            if one_line.count(error_word) == 1:
                errorfre_list.append(error_word)
            
            #여기 잘 이해 안 됨...
            elif one_line.count(error_word) > 1:
                i = error_word * one_line.count(error_word)
                for x in list(i):
                    #print(x)
                    errorfre_list.append(x)

            #(추가) 한글 사전 탐색
            #for one_word in one_line:
             #   if one_word not in self.korean_dict_list:
              #      errorfre_list.append(one_word)

        f.close()



        return cnt, errorfre_list
