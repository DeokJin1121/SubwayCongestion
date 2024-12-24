import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('filtered_subway_data.csv')

# "호선" 열의 값이 "5호선"인 행 필터링
filtered_df = df[df['호선'] == '5호선']

# "연번" 열을 1부터 시작하는 오름차순으로 재설정
filtered_df['연번'] = range(1, len(filtered_df) + 1)

# 필터링된 데이터프레임을 새로운 CSV 파일로 저장
filtered_df.to_csv('filtered_train_time.csv', index=False)
