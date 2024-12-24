import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 데이터 로드 및 전처리
data = pd.read_csv('C:\\GitHub\\BigDataProject\\5train_data.csv', encoding='utf-8')

# 컬럼 이름 설정
columns = [
    '요일', '호선', '역번호', '역명', '구분', '06시 이전', '06시-07시', '07시-08시', 
    '08시-09시', '09시-10시', '10시-11시', '11시-12시', '12시-13시', '13시-14시', 
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
    return congestion_prediction[0]

# 예시: 사용자 입력 기반 예측
boarding_station_name = input("탑승역명을 입력하세요: ")
day = input("요일을 입력하세요 (예: '월요일'): ")
time = int(input("시간대를 입력하세요 (0-23): "))

predicted_congestion = predict_congestion(boarding_station_name, day, time)
print(f"예측된 혼잡도: {predicted_congestion}")
