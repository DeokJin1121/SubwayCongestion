from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import io
import base64

# 데이터 로드 및 전처리
data = pd.read_csv('C:\\GitHub\\BigDataProject\\subway_congestion\\data\\5train_data.csv', encoding='utf-8')
columns = [
    '요일', '호선', '역번호', '역명', '구분', '6시 이전', '6시-7시', '7시-8시', 
    '8시-9시', '9시-10시', '10시-11시', '11시-12시', '12시-13시', '13시-14시', 
    '14시-15시', '15시-16시', '16시-17시', '17시-18시', '18시-19시', '19시-20시', 
    '20시-21시', '21시-22시', '22시-23시', '23시-24시', '24시 이후'
]
data.columns = columns

station_mapping = data[['역명', '역번호']].drop_duplicates().set_index('역명')['역번호'].to_dict()
data_ride = data[data['구분'] == '승차']
data_alight = data[data['구분'] == '하차']

def calculate_congestion(data_ride, data_alight, time_slots):
    congestion_data = []
    for time_slot in time_slots:
        ride_counts = data_ride.groupby(['역번호', '요일'])[time_slot].sum().reset_index()
        alight_counts = data_alight.groupby(['역번호', '요일'])[time_slot].sum().reset_index()
        congestion = ride_counts.merge(alight_counts, on=['역번호', '요일'], suffixes=('_ride', '_alight'))
        congestion['congestion'] = congestion[f'{time_slot}_ride'] - congestion[f'{time_slot}_alight']
        congestion['time_slot'] = time_slot
        congestion_data.append(congestion)
    return pd.concat(congestion_data)

time_slots = columns[5:]
congestion_data = calculate_congestion(data_ride, data_alight, time_slots)

X = congestion_data[['역번호', '요일', 'time_slot']]
y = congestion_data['congestion']
X = pd.get_dummies(X, columns=['요일', 'time_slot'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

def predict_congestion(boarding_station_name, day, time):
    if boarding_station_name not in station_mapping:
        raise ValueError(f"역명 '{boarding_station_name}'을(를) 찾을 수 없습니다.")
    
    boarding_station = station_mapping[boarding_station_name]
    time_slot = f'{time}시-{time+1}시' if time < 24 else '24시 이후'
    
    input_data = pd.DataFrame({
        '역번호': [boarding_station],
        '요일': [day],
        'time_slot': [time_slot]
    })
    input_data = pd.get_dummies(input_data, columns=['요일', 'time_slot'])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    congestion_prediction = model.predict(input_data)
    return congestion_prediction[0] / 13

def index(request):
    stations = list(station_mapping.keys())
    return render(request, 'index.html', {'stations': stations})

def predict(request):
    if request.method == 'POST':
        boarding_station_name = request.POST.get('boarding_station')
        day = request.POST.get('day')
        time_slot = request.POST.get('time_slot')
        time = int(time_slot.split('-')[0].replace('시', '')) if '이전' not in time_slot else 0
        
        predicted_congestion_values = []
        for t in range(25):
            predicted_congestion = predict_congestion(boarding_station_name, day, t)
            predicted_congestion_values.append(predicted_congestion)
        
        # 마지막 예측된 혼잡도 값을 기준으로 혼잡도 상태를 결정합니다.
        last_predicted_congestion = predicted_congestion_values[-1]
        if last_predicted_congestion >= 120:
            congestion = '매우 혼잡'
        elif last_predicted_congestion >= 100:
            congestion = '혼잡'
        elif last_predicted_congestion >= 70:
            congestion = '보통'
        elif last_predicted_congestion >= 50:
            congestion = '여유'
        else:
            congestion = '매우 여유'
        
        ride_count = data_ride[(data_ride['역번호'] == station_mapping[boarding_station_name]) & (data_ride['요일'] == day)][time_slot].sum()
        alight_count = data_alight[(data_alight['역번호'] == station_mapping[boarding_station_name]) & (data_alight['요일'] == day)][time_slot].sum()

        # 한글 폰트 경로 설정
        font_path = 'C:\\Windows\\Fonts\\batang.ttc'  # 사용하고 있는 환경에 맞는 한글 폰트 경로로 수정해주세요
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)

        plt.figure(figsize=(10, 6))
        plt.plot(range(25), predicted_congestion_values, marker='o', linestyle='-', color='b')
        plt.title(f'{day}의 {boarding_station_name}역 혼잡도 예측')
        plt.xlabel('시간대')
        plt.ylabel('예측 혼잡도')
        plt.xticks(np.arange(0, 25, step=1))
        plt.grid(True)

        plt.text(0.88, 1.1, f'{time}시 승차자 수: {int(ride_count / 13)}',
             horizontalalignment= 'center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),
        plt.text(0.88, 1.05, f'{time}시 하차자 수: {int(alight_count / 13)}',
             horizontalalignment= 'center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),

        plt.text(0.03, 1.1, '50 = 여유   70 = 보통',
            horizontalalignment= 'left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),
        plt.text(0.03, 1.05, '100 = 혼잡   120 = 매우 혼잡',
            horizontalalignment= 'left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        
        return render(request, 'result.html', {
            'image_base64': image_base64,
            'boarding_station_name': boarding_station_name,
            'day': day,
            'time': time,
            'ride_count': int(ride_count / 13),
            'alight_count': int(alight_count / 13),
            'congestion': congestion
            
        })
    return HttpResponse("Invalid request method", status=405)
