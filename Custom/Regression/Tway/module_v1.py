import pandas as pd
import numpy as np

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

    