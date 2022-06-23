# 바이러스토탈 URL 스캔 주소
import time, requests

my_apikey = '2e1151e1ef625c5cae75901f14d821bfe573323e06dc480dd76e9b22761d6ac0'
url_scan = 'https://www.virustotal.com/vtapi/v2/url/scan' #바이러스 토탈 주소

def url_check(my_url):
    scan_params = {'apikey': my_apikey, 'url': my_url}
    scan_response = requests.post(url_scan, data = scan_params)
    result_scan = scan_response.json()

    scan_id = result_scan['scan_id']

    # URL 스캔 시작
    print('Virustotal URL SCAN START : ', my_url, '\n')

    # 바이러스토탈 URL 점검 결과 주소
    url_report = 'https://www.virustotal.com/vtapi/v2/url/report'
    report_params = {'apikey': my_apikey, 'resource': scan_id } #my_url
    report_response = requests.get(url_report, params=report_params)

    # 점검 결과 데이터 추출
    report = report_response.json()
    report_scan_date = report.get('scan_date')
    report_scan_result = report.get('scans')
    report_scan_venders = list(report['scans'].keys())

    # URL 점검 결과 리포트 조회하기
    # 점검 완료 메시지
    print(report.get('verbose_msg'), '\n')
    time.sleep(1)

    # 스캔 결과 데이터만 보기
    print('Scan Date (UTC) : ', report_scan_date)

    for vender in report_scan_venders:
        outputs = report_scan_result[vender]
        outputs_keys = report_scan_result[vender].get('result')

    re_ = report_scan_result["Google Safebrowsing"]["result"]

    if re_ == "clean site" :
        #안전
        #print("0")
        return 0
    else:
        #위험/수상
        #print("1")
        return 1
