from selenium import webdriver
import time
import pandas as pd
import csv


###########네이버 뉴스 댓글 가져오는 함수##############
#url="https://news.naver.com/main/read.nhn?mode=LSD&sid1=001&oid=023&aid=0003547854"
def capture_replys_naver(url,excel_name):
    if not '&m_view=1' in url:
        url += '&m_view=1'
    #드라이버 연결 인스턴스
    driver = webdriver.Chrome("chromedriver")
    driver.implicitly_wait(1)
    driver.maximize_window()

    #페이지 이동
    print('[접속하기]')
    driver.get(url)

    #더 보기 버튼 누르기
    print('[더 보기 클릭 중]')
    attemp=0 # 더 보기 탐색 횟수
    while True:
        try:
            driver.find_element_by_css_selector('span.u_cbox_page_more').click()
            attemp = 0
        except:
            attemp +=1
            if attemp >5:
                break  # 더 이상 더 보기 버튼이 없다라고 생각하겠다.

    ##댓글 요소 찾기    
    print('[댓글 요소 찾기]')
    replys = driver.find_elements_by_css_selector('ul.u_cbox_list > li.u_cbox_comment')
    print("전체 댓글 개수:" ,len(replys))

    # 작성자: span.u_cbox_nick, 댓글 내용:span.u_cbox_contents
    ##작성자와 댓글 내용 추출(작성자, 댓글 내용)
    print('[댓글 내용 수집]')
    results = []
    del_msg = 0
    for reply in replys:
        try:
            # author = reply.find_element_by_css_selector("span.u_cbox_nick").text
            content =  reply.find_element_by_css_selector("span.u_cbox_contents").text
            # results.append((author,content))
            results.append(content)
        except:
            del_msg +=1

    print('삭제된 댓글 개수:', del_msg)

    ##엑셀 파일 만들기
    print('[전체 댓글 엑셀로 저장]')
    # cols = ['작성자', '내용']
    # data_frame = pd.DataFrame(results, columns =cols)
    data_frame = pd.DataFrame(results)
    data_frame.to_csv('C:/source_code/tensorflow/bigtering_202.10.10/excelfile/{}.csv'.format(excel_name), mode="a", header=False, index=False, encoding='utf-8-sig')

    #닫아주기
    time.sleep(5)
    driver.quit()