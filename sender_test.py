import json
import urllib.request
import re
import spam_predict

#웹문서 제목 가져오기

def sender_check(number):
    client_id = "Kc1I24mmVCGnd_4H1Jg8"
    client_secret = "pA6dOeRage"

    encText = urllib.parse.quote(number) # 한글을 URL에 추가하기 위해서 UTF-8 형식으로 URL 인코딩

    url = "https://openapi.naver.com/v1/search/webkr?display=3&query=" + encText # json 결과

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)

    response = urllib.request.urlopen(request) #url 가져오기
    rescode = response.getcode()

    #내용 전체
    if(rescode==200):
        response_body = response.read()
        #print(response_body.decode('utf-8'))

    else:
        print("Error Code:" + rescode)

    #제목만
    text_data = response_body.decode('utf-8')
    json_data = json.loads(text_data)

    #print(json_data)

    for x in json_data['items']:
        result = re.sub('<.+?>', '', x['title'],0,re.I | re.S)
        if number in result:
            print("결과 : " + result)

            # 인공지능으로 result 전송
            A_result = spam_predict.spam_predict_se(result)
            print("--------------" + str(A_result))
            return A_result

        else :
            result = 0
            return result

    #sender_re = AI_sender.AI_number()
    #print(sender_re)

