""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 데이터 로드 및 전처리
data = pd.read_csv('C:\\GitHub\\BigDataProject\\5train_data.csv', encoding='utf-8')

# 컬럼 이름 설정
columns = [
    '요일', '호선', '역번호', '역명', '구분', '6시 이전', '6시-7시', '7시-8시', 
    '8시-9시', '9시-10시', '10시-11시', '11시-12시', '12시-13시', '13시-14시', 
    '14시-15시', '15시-16시', '16시-17시', '17시-18시', '18시-19시', '19시-20시', 
    '20시-21시', '21시-22시', '22시-23시', '23시-24시', '24시 이후'
]
data.columns = columns

# 역명과 역번호 매핑 생성
station_mapping = data[['역명', '역번호']].drop_duplicates().set_index('역명')['역번호'].to_dict()

# 승차자 수와 하차자 수를 구분하여 처리
data_ride = data[data['구분'] == '승차']
data_alight = data[data['구분'] == '하차']

# 각 시간대별로 승차자 수와 하차자 수를 계산하여 혼잡도를 구함
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

# 예시: 모든 시간대에 대한 혼잡도 계산
time_slots = columns[5:]  # 시간대 컬럼 선택
congestion_data = calculate_congestion(data_ride, data_alight, time_slots)

# 예측 모델 구축
X = congestion_data[['역번호', '요일', 'time_slot']]
y = congestion_data['congestion']

# 카테고리 특성(요일, 시간대) 인코딩
X = pd.get_dummies(X, columns=['요일', 'time_slot'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# 사용자 입력 기반 예측 함수
def predict_congestion(boarding_station_name, day, time):
    # 역명 -> 역번호 매핑
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
    input_data = input_data.reindex(columns=X.columns, fill_value=0)  # 기존 X의 컬럼과 맞추기

    congestion_prediction = model.predict(input_data)
    return congestion_prediction[0] / 13

# 예시: 사용자 입력 기반 예측
boarding_station_name = input("탑승역명을 입력하세요: ")
day = input("요일을 입력하세요 (예: '월요일'): ")
time = int(input("시간대를 입력하세요 (0-23): "))

# 승차자 수와 하차자 수 출력
boarding_station = station_mapping.get(boarding_station_name)
if boarding_station is not None:
    time_slot = f'{time}시-{time+1}시' if time < 24 else '24시 이후'
    ride_count = data_ride[(data_ride['역번호'] == boarding_station) & (data_ride['요일'] == day)][time_slot].sum()
    alight_count = data_alight[(data_alight['역번호'] == boarding_station) & (data_alight['요일'] == day)][time_slot].sum()
    print(f"{day} 0{time}시의 '{boarding_station_name}' 역의 승차자 수: {ride_count / 13}, 하차자 수: {alight_count / 13}")

    predicted_congestion = predict_congestion(boarding_station_name, day, time)
    #calculated_congestion = (predict_congestion / 13.0) # 13주의 데이터 셋
    print(f"예측된 혼잡도: {predicted_congestion / 13.0}")
    #print(f"계산된 혼잡도: {calculate_congestion}")
else:
    print(f"입력한 '{boarding_station_name}' 역명은 존재하지 않습니다.")

# 한글 폰트 경로 설정
font_path = 'C:\\Windows\\Fonts\\batang.ttc'  # 사용하고 있는 환경에 맞는 한글 폰트 경로로 수정해주세요
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 사용자 입력에 대한 예측
predicted_congestion_values = []

# 0부터 24까지의 시간대에 대해 혼잡도 예측
for t in range(25):
    predicted_congestion = predict_congestion(boarding_station_name, day, t)
    predicted_congestion_values.append(int(predicted_congestion / 13))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(25), predicted_congestion_values, marker='o', linestyle='-', color='b')
plt.title(f'{day}의 {boarding_station_name}역 혼잡도 예측')
plt.xlabel('시간대')
plt.ylabel('예측 혼잡도')
plt.xticks(np.arange(0, 25, step=1))  # x축 눈금을 1단위로 표시
plt.grid(True)

plt.text(0.88, 1.1, f'{time}시 승차자 수: {int(ride_count / 13)}',
             horizontalalignment= 'center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),
plt.text(0.88, 1.05, f'{time}시 하차자 수: {int(alight_count / 13)}',
             horizontalalignment= 'center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),

plt.text(0.03, 1.1, '50 = 여유   70 = 보통',
    horizontalalignment= 'left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),
plt.text(0.03, 1.05, '100 = 혼잡   120 = 매우 혼잡',
    horizontalalignment= 'left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=10, fontweight='bold'),

plt.show() """