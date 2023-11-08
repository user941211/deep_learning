import os
import pandas as pd
from datetime import datetime, timedelta


# 이 파일은 학습이 완료되어서 predict라는 폴더에 들어간 csv 파일들에 파일 목록들을
# 각각 date 컬럼을 추가하여 10월 20까지만 있는 기존 칼럼 내용에서 추가로 예측한 날짜만큼 추가하는 파일입니다.

# 폴더 경로 설정
directory_path = "C:/Users/mycom/Desktop/predict"  

# 시작 날짜 설정
start_date = datetime(2023, 10, 21)

# 폴더 내의 모든 CSV 파일 목록을 가져옵니다.
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# 각 CSV 파일에 대해 작업 수행
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    
    # CSV 파일을 읽어옵니다.
    # 인코딩은 여전히 ANSI 이기 때문에 cp949로 진행합니다.
    stock_data = pd.read_csv(file_path, encoding='cp949')
    
    # 'date' 컬럼 생성 및 날짜 값 할당
    num_rows = len(stock_data)
    stock_data['date'] = [start_date + timedelta(days=i) for i in range(num_rows)]
    
    # 수정된 데이터프레임을 기존 CSV 파일로 저장 (덮어쓰기)
    stock_data.to_csv(file_path, index=False, encoding='cp949')

    print(f"Added 'date' column to {file} in {file_path}")

print("모든 파일에 대한 작업 완료")