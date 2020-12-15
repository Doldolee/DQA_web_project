from selenium import webdriver
import time
import pandas as pd


################다음 뉴스 댓글 가져오는 함수################
#url= 'https://news.v.daum.net/v/20200725152101943'
def capture_replys_daum(url, excel_name):

    driver = webdriver.Chrome('chromedriver')
    driver.implicitly_wait(3)
    driver.maximize_window()

    #다음 페이지 이동
    print('[접속하기]')
    driver.get(url)


    #과거순 댓글 클릭하기
    print('[과거순 댓글 클릭하기]')
    driver.find_element_by_xpath('//*[@id="alex-area"]/div/div/div/div[3]/ul[1]/li[4]/button/span/span[1]').click()
    time.sleep(2)


    #더 보기 클릭하기-1
    print('[더 보기 클릭하기]')
    attempt=0
    while True:
        try:
            driver.find_element_by_xpath('//*[@id="alex-area"]/div/div/div/div[3]/div[2]/button/span[1]').click()
            attempt=0
            time.sleep(1)
        except:
            attempt+=1
            if attempt>5:
                break
    time.sleep(2)

     #전체 댓글 갯수
    print('[전체 댓글 갯수]')
    replys = driver.find_element_by_css_selector('#alex-header > em').text
    review_count = int(replys)
    print('전체 댓글 수 :', review_count)
    time.sleep(2)

    # #댓글 본문 가져오기
    print('[댓글 본문 가져오기]')
    results=[]
    del_msg=0
    replys = driver.find_elements_by_css_selector("ul.list_comment > li")
    for reply in replys:
        try:
            content = reply.find_element_by_css_selector("p.desc_txt").text
            content = content.replace("\n", " ")
            results.append(content)
        except:
            del_msg +=1
    time.sleep(2)
    print('삭제된 댓글: ', review_count-len(results))

    # #엑셀 파일로 저장
    data_frame=pd.DataFrame(results)
    data_frame.to_excel('C:/source_code/tensorflow/bigtering_202.10.10/excelfile/{}.xlsx'.format(excel_name), index=False, sheet_name='뉴스')

    #드라이버 종료
    time.sleep(2)
    driver.quit()
    