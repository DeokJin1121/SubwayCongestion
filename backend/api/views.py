from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import joblib
import os

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            time = data.get('시간')
            station = data.get('역명')

            if not time or not station:
                return Response({'error': '시간과 역명을 모두 제공해야 합니다.'}, status=status.HTTP_400_BAD_REQUEST)

            # 모델과 컬럼 로드
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
            columns_path = os.path.join(os.path.dirname(__file__), '..', 'model_columns.pkl')
            model = joblib.load(model_path)
            model_columns = joblib.load(columns_path)

            # 입력 데이터를 DataFrame으로 변환
            input_data = pd.DataFrame([[time, station]], columns=['시간', '역명'])
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=model_columns, fill_value=0)

            # 예측
            prediction = model.predict(input_data)
            return Response({'혼잡도': prediction[0]})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
