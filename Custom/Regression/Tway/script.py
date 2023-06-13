import argparse
import joblib
import pickle
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from contextlib import suppress
import awswrangler as wr
import datetime
import boto3

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("extracting arguments")    
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--total", type=str, default=os.environ.get("SM_CHANNEL_TOTAL"))
    parser.add_argument("--total-file", type=str, default="train_data.parquet")
    parser.add_argument("--features", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--key", type=str)
    
    args, _ = parser.parse_known_args()
    
    bucket = args.bucket
    key = args.key

    ## Preprocessing 2
    ## Load coupon & booking Data
    a, b, c, year, month, filename = key.split('/')
    end_date = datetime.datetime.strptime(f'{year}-{int(month) + 1}','%Y-%m') - datetime.timedelta(days=1)
    ## 데이터 기간 설정
    start_date = end_date - datetime.timedelta(days=3650)
    date_range = pd.date_range(start_date, end_date, freq= 'M')
    
    coupon_list = []
    booking_list = []
    
    for idx, date in enumerate(date_range):
        if len(str(date.month)) == 1 :
            temp_month = '0'+str(date.month)
        else:
            temp_month = date.month
            
        with suppress(Exception): coupon_list.append(wr.s3.read_parquet(f"s3://{bucket}/train_data/coupon/pps_data/{date.year}/{temp_month}/coupon_df.parquet"))
        with suppress(Exception): booking_list.append(wr.s3.read_parquet(f"s3://{bucket}/train_data/booking/pps_data/{date.year}/{temp_month}/booking_df.parquet"))
    
    coupon_df = pd.concat(coupon_list)
    booking_df = pd.concat(booking_list)
   
    ## Merge coupon_df & booking_df
    orig_df = pd.merge(booking_df, coupon_df, on=list(set(list(booking_df)) & set(list(coupon_df))), how = 'right')
    orig_df.dropna(axis = 0, inplace = True)

    ## Load flight Data
    print(orig_df['flight_departure_date'].min(),orig_df['flight_departure_date'].max())
    flight_range = pd.date_range(orig_df['flight_departure_date'].min(),orig_df['flight_departure_date'].max())
    flight_df = pd.DataFrame()
    for date in flight_range:
        if len(str(date.month)) == 1 :
            temp_month = '0'+str(date.month)
        else:
            temp_month = date.month
        if len(str(date.day)) == 1 :
            temp_day = '0'+str(date.day)
        else:
            temp_day = date.day
        with suppress(Exception): flight_df = pd.concat([flight_df, wr.s3.read_parquet(f"s3://{bucket}/train_data/flight/pps_data/{date.year}/{temp_month}/{temp_day}/flight_df.parquet")])
    
    flight_raw = flight_df.shape
    coupon_raw = coupon_df.shape
    booking_raw = booking_df.shape

    ## Drop Duplicates
    coupon_df.drop_duplicates(inplace = True)
    booking_df.drop_duplicates(inplace = True)
    flight_df.drop_duplicates(inplace = True)
    
    ## Merge coupon_df & booking_df & flight_df
    orig_df = pd.merge(flight_df, orig_df, on = list(set(list(flight_df)) & set(list(orig_df))), how = 'right')
    orig_df.reset_index(drop = True, inplace = True)
    
    ## Create Columns
    total_df = module_v1.create_columns(orig_df, booking_df)
    total_df.dropna(axis = 0, inplace = True)

    ## Save Data
    # wr.s3.to_parquet(df = total_df,
    #                  path=f"s3://{bucket}/train_data/{datetime.datetime.today().strftime('%Y-%m-%d %H')}/train_data.parquet")   
    
    ## Data Report
    sns_arn = os.environ['snsARN']
    snsclient = boto3.client('sns')
    try:
        message = ""
        message += "\nLambda Data Report22222222" + "\n\n"
        message += "############################################################################\n"
        message += "# Data PreProcessing & Train Dataset Save Done "+ "\n"
        message += "# Coupon Shape : " + str(coupon_raw) + " -> Drop Duplicates Shape : " + str(coupon_df.shape) + "\n"
        message += "# Booking Shape : " + str(booking_raw) + " -> Drop Duplicates Shape : " + str(booking_df.shape) + "\n"
        message += "# Flight Shape : " + str(flight_raw) + " -> Drop Duplicates Shape : " + str(flight_df.shape) + "\n\n"
        message += "# All Join Data Shape : " + str(orig_df.shape) + " -> After PreProcessing 2 Shape : " + str(total_df.shape) + "\n"
        message += "# Data Example : \n" + str(orig_df.head()) +"\n"
        message += "############################################################################\n"
    
        snsclient.publish(
            TargetArn=sns_arn,
            Subject=f'Lambda Data Report',
            Message=message
        )
    except Exception as e:
        print(e)
    
    # total_df = pd.read_parquet(os.path.join(args.total, args.total_file))
    
    print("building training and testing datasets")
    total_x = total_df[args.features.split()]
    total_y = total_df[args.target]
    
    train_x, test_x, train_y, test_y = train_test_split(total_x, total_y, test_size=0.2, random_state=42)

    # train
    print("training model")
    model = RandomForestRegressor(random_state=42)
    model.fit(train_x, train_y)

    # print abs error
    print("validating model")
    abs_err = np.abs(model.predict(test_x) - test_y)

    # print couple perf metrics
    for q in [10, 50, 90]:
        print("AE-at-" + str(q) + "th-percentile: " + str(np.percentile(a=abs_err, q=q)))

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)

    
def create_columns(orig_df, booking_df):
    ## Create Columns (누적예약 수)
    seats_df = pd.DataFrame({'total_sold_seats' : booking_df.groupby(["flight_number", "capture_date", 'cabin_code', 'flight_departure_date'])['sold_seats'].sum()}).reset_index()  
    total_df = pd.merge(seats_df, orig_df, on = list(set(list(seats_df)) & set(list(orig_df))), how = 'right')
    
    ## Create Columns About Time
    total_df['remain_seats'] = total_df['cabin_authorised_cap'] - total_df['total_sold_seats'] ## 전일 잔여 좌석 수
    total_df['departure_hour'] = total_df['leg_departure_date_time'].dt.hour ## 출발 시간
    total_df['departure_minute'] = total_df['leg_departure_date_time'].dt.minute ## 출발 분
    total_df['departure_time_format_min'] = total_df['departure_hour'] * 60 + total_df['departure_minute'] ## 출발 시간+분 -> 분
    total_df['issueweekday'] = total_df['capture_date'].dt.weekday ## 티켓 구매 요일
    total_df['departureweekday'] = total_df['leg_departure_date_time'].dt.weekday ## 항공권 출발 요일
    total_df['remain_days'] = total_df['flight_departure_date'] - total_df['capture_date'] ## 출발까지 잔여일
    total_df['remain_days'] = total_df['remain_days'].astype('str').str.split().str[0]
    total_df['remain_days'] = total_df['remain_days'].astype('int')
    
    dummy_list = ['departureweekday', 'issueweekday','startseg']
    for i in dummy_list:
        dummy_data = pd.get_dummies(total_df[i])        
        dummy_data = pd.DataFrame(columns = [f'{i}_{j}' for j in list(dummy_data)], data = dummy_data.values)
        dummy_data.reset_index(drop = True, inplace = True)
        total_df = pd.merge(total_df, dummy_data, right_index = True, left_index = True)
        total_df.reset_index(drop = True, inplace = True)
        
    return total_df