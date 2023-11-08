import os
import pandas as pd

# 이 파일은 폴더 내부에 있는 모든 엑셀들의 제목을 변경하는 코드입니다.
# DB에서 엑셀로 export를 했을 때, ANSI로 export를 하니 엑셀 내부에 한글은 깨지지 않는데
# 한글인 제목은 깨졌기 때문에, 종목명의 첫번째 값을 엑셀 제목으로 바꾸는 코드입니다.

# 엑셀 파일이 저장된 폴더 경로
folder_path = "C:/Users/commo/Desktop/project/db 내용"

# 폴더 내의 모든 파일 목록을 가져옵니다.
file_list = os.listdir(folder_path)

# ".xlsx" 확장자를 가진 파일만 고려하도록 필터링
xlsx_files = [f for f in file_list if f.endswith(".xlsx")]

for xlsx_file in xlsx_files:
    # 엑셀 파일 경로
    excel_file_path = os.path.join(folder_path, xlsx_file)

    # 엑셀 파일을 DataFrame으로 읽기
    df = pd.read_excel(excel_file_path)

    # code_name 컬럼의 첫 번째 값 가져오기
    first_code_name = df['code_name'].iloc[0]

    # 엑셀 파일의 제목을 code_name으로 업데이트
    os.rename(excel_file_path, os.path.join(folder_path, f"{first_code_name}.xlsx"))

    print(f"File '{xlsx_file}' renamed to '{first_code_name}.xlsx'")