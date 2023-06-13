import json
import boto3
import joblib
import tarfile
import pandas as pd
import os
import datetime

def handler(event, context):
    # TODO implement
    print('start!')
    s3_client = boto3.client('s3')
    bucket = 'poc-2209-twayairport-dp'
    prefix = 'sagemaker-sklearn-artifact'
    
    obj_list = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    obj_list = [i['Key'].split('/')[1] for i in obj_list['Contents'] if 'output' in i['Key']]
    print('=' * 70)
    print(obj_list)
    print(max(obj_list))
    model_file_nm = max(obj_list)    
    print('=' * 70)
    
    s3_client.download_file(bucket, f'{prefix}/{model_file_nm}/output/model.tar.gz', '/tmp/model.tar.gz')
    
    file = tarfile.open('/tmp/model.tar.gz')
    file.extractall('/tmp')
    file.close()
    
    path = '/tmp'
    file_list = os.listdir(path)
    print(f'file_list {file_list}')
    
    model = joblib.load('/tmp/model.joblib')
    print(model)

    ## Data Preprocessing
    capture_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    ## 수정 필요
    leg_departure_date_time = '2022-12-31 13:25:00'
    startseg = 'GMP'
    total_sold_seats = 31
    cabin_authorised_cap = 189

    if startseg == 'GMP': startseg_GMP, startseg_CJU = 1, 0
    else: startseg_GMP, startseg_CJU = 0,1
    remain_seats = int(cabin_authorised_cap) - int(total_sold_seats)

    ## Create Columns About Time
    leg_departure_date_time = datetime.datetime.strptime(leg_departure_date_time,'%Y-%m-%d %H:%M:%S')
    departure_hour = leg_departure_date_time.hour ## 출발 시간
    departure_minute = leg_departure_date_time.minute ## 출발 분
    departure_time_format_min = departure_hour * 60 + departure_minute ## 출발 시간+분 -> 분

    capture_date = datetime.datetime.strptime(capture_date,'%Y-%m-%d')
    issueweekday = capture_date.weekday() ## 티켓 구매 요일
    departureweekday = leg_departure_date_time.weekday() ## 항공권 출발 요일
    remain_days = leg_departure_date_time - capture_date ## 출발까지 잔여일
    remain_days = remain_days.days

    for i in range(7):
        globals()[f'departureweekday_{i}'] = 0
        globals()[f'issueweekday_{i}'] = 0
        if i == departureweekday: globals()[f'departureweekday_{i}'] = 1
        if i == issueweekday: globals()[f'issueweekday_{i}'] = 1   
        
    features = ['total_sold_seats', 'startseg_CJU', 'startseg_GMP', 'remain_seats', 'departure_hour', 'departure_minute', 'departure_time_format_min', 'remain_days', 
                'departureweekday_0', 'departureweekday_1', 'departureweekday_2', 'departureweekday_3', 'departureweekday_4', 'departureweekday_5', 'departureweekday_6', 
                'issueweekday_0', 'issueweekday_1', 'issueweekday_2', 'issueweekday_3', 'issueweekday_4', 'issueweekday_5', 'issueweekday_6']      
    df = pd.DataFrame(columns = features, index = [0])
    for i in list(df):
        df[i] = globals()[i]
        
    print(model.predict(df))
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
