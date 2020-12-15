from openpyxl import load_workbook
from jamo import h2j, j2hcj
import csv

# # 엑셀 리스트로 읽어오는 함수
# wb = load_workbook("comments.xlsx", read_only=True)
# data1 = wb.active
# script = []
# for row in data1:
#     for cell in row:
#         var1 = cell.value
#         script.append(var1)
#         script[0] = "comments"


sentence1 = "10월백신출싥~~현실이ㅜㅜ 되면 좋겠네요~~"

#띄어쓰기 개수 세기
def count_space(sentence):
    return sentence.count(' ')
#단모음 + 단자음 개수 세기
def count_single_letter(sentence):
    ja = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㄳ','ㄵ','ㄶ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅄ']
    mo = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
    cnt = 0
    for ch in sentence:
        if(ch in ja or ch in mo):
            cnt += 1
    return cnt

# while True:
#     sentence = input("문장 입력:")
#     if sentence == "":
#         break

# 1개 문장에서 자음/모음/특수문자/숫자 개수 리턴하는 함수
def count_con_vow_num_spe(sentence):
  sentence = j2hcj(h2j(sentence))
  # print(sentence)

  # 초성 리스트
  CHOSUNG_LIST=['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
  # 중성 리스트
  JUNGSUNG_LIST=['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
  # 종성 리스트
  JONGSUNG_LIST=['ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
  # 숫자 리스트
  NUMBER_LIST=['0','1','2','3','4','5','6','7','8','9']
  # 특수문자 리스트
  SPECIAL_LIST=['~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','+','-','`',';',"'",':','>','<','?','/']

  count_consonant=[]
  count_vowel=[]
  count_number=[]
  count_special=[]

  for word in sentence:
    if word in CHOSUNG_LIST or word in JONGSUNG_LIST:
      count_consonant.append(word)
    elif word in JUNGSUNG_LIST:
      count_vowel.append(word)
    elif word in NUMBER_LIST:
      count_number.append(word)
    elif word in SPECIAL_LIST:
      count_special.append(word)

  count_consonant =len(count_consonant)
  count_vowel =len(count_vowel)
  count_number =len(count_number)
  count_special =len(count_special)
  
  return count_consonant, count_vowel, count_number, count_special

# 반환 값
# count_consonant = count_con_vow_num_spe(sentence1)[0]
# count_vowel = count_con_vow_num_spe(sentence1)[1]
# count_number = count_con_vow_num_spe(sentence1)[2]
# count_special = count_con_vow_num_spe(sentence1)[3]


# print("띄어쓰기 개수:", count_space(sentence1))
# print("단자음 단모음 개수:", count_single_letter(sentence1))
# print("자음 개수", count_consonant)
# print('모음 개수', count_vowel)
# print('숫자 개수', count_number)
# print('특수 문자 개수', count_special)

dataInput_excel=[]
script2=['나는 바보입1니당!!!','너ㅚㅇㄴㄹㄵㄷㄹ34ㅈㄹ두 사 랑 다']
for sentence2 in script2:
  count_space1 = count_space(sentence2)
  count_single_letter1 = count_single_letter(sentence2)
  count_consonant1 = count_con_vow_num_spe(sentence2)[0]
  count_vowel1 = count_con_vow_num_spe(sentence2)[1]
  count_number1 = count_con_vow_num_spe(sentence2)[2]
  count_special1 = count_con_vow_num_spe(sentence2)[3]
  
  list1 = [ sentence2, count_space1, count_single_letter1, count_consonant1,count_vowel1,count_number1,count_special1]
  count_space1=''
  count_single_letter1=''
  count_consonant1=''
  count_vowel1=''
  count_number1=''
  count_special1=''
  dataInput_excel.append(list1)
print(dataInput_excel)

#엑셀파일 x인자
with open('test.csv', 'w' ,newline='') as f:
  makewrite = csv.writer(f)
  index=['단어','띄어쓰기','단모음단자음','자음','모음','숫자','특수문자']
  makewrite.writerow(index)
  for value in dataInput_excel:
    makewrite.writerow(value)
