from flask import Flask, render_template, request, session, redirect
import csv
import requests
from bs4 import BeautifulSoup as bs
import bs4
from openpyxl import Workbook
from itertools import chain
from openpyxl import load_workbook
from wordcloud import WordCloud
import re
from konlpy.tag import Okt
from functools import wraps
import pymongo
from pymongo import MongoClient
from datetime import date, timedelta
from collections import Counter
from function import crawler 
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')



okt = Okt()

app = Flask(__name__)
app.secret_key = b"7\xe39,\x8eE\xff\xe5\x9b\x95\xc6,\xfcX'b"

#database
cluster = MongoClient("mongodb+srv://sangbeen:12341234@boilerplate.cw0as.gcp.mongodb.net/user_login_system?retryWrites=true&w=majority")
db = cluster["user_login_system"]
collection = db["user"]


#Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')
    return wrap 

# Route
from user import routes

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/dashboard/')
@login_required
def dashboard():
    return render_template('main.html')


'''
    #국어사전 추가시 필요
    excel_data_df = pd.read_excel('korean_dict.xlsx', sheet_name='NIADic')
    korean_dict_list = excel_data_df['term'].tolist()
'''
#@app.route('/user/signup', methods=['POST'])

@app.route('/post', methods=['POST'])
def crawl():
    return crawler.Crawler().crawl()


if __name__ =="__main__":
  app.run(debug=True)
