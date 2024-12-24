import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 데이터 파일 읽기
file_path = 'C:\\Users\\JangYeonWoo\\Desktop\\MachineLearning\\TermProject\\train_time.csv'
data = pd.read_csv(file_path, encoding='euc-kr')

# 데이터 확인
print(data.head())

# '혼잡도' 계산
data['혼잡도'] = data.apply(lambda row: '높음' if row['승차자 수'] > (row['승차자 수'] + row['하차자 수']) * 0.8 else '낮음', axis=1)

# '시간', '역명'을 특징으로 설정
X = data[['시간', '역명']]
y = data['혼잡도']

# '역명'을 숫자로 변환 (one-hot encoding)
X = pd.get_dummies(X, columns=['역명'])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 평가
print(f'Accuracy: {model.score(X_test, y_test)}')

# 모델과 컬럼 저장
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
