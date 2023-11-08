import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# 이 파일은 DATA GRIP을 통해서 ANSI로 인코딩 된 엑셀을 엑셀 파일들을 csv로 변환하는 코드입니다.

def convert_excel_to_csv(input_file, output_folder):
    try:
        df = pd.read_excel(input_file)
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '.csv')
        df.to_csv(output_file, index=False, encoding='ANSI')
        print(f'변환 완료: {os.path.basename(input_file)} -> {os.path.basename(output_file)}')
    except Exception as e:
        print(f'변환 중 오류 발생: {os.path.basename(input_file)} - {str(e)}')

if __name__ == "__main__":
    input_folder = 'C:/Users/commo/Desktop/project/db 내용'
    output_folder = 'C:/Users/commo/Desktop/project/db_csv'

    if not os.path.exists(input_folder):
        print(f'입력 폴더가 존재하지 않습니다: {input_folder}')
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'출력 폴더가 생성되었습니다: {output_folder}')

    # 입력 폴더 내의 XLSX 파일 목록 가져오기
    input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.xlsx')]

    # 멀티 스레드를 사용하여 파일 변환 작업 실행
    with ThreadPoolExecutor(max_workers=8) as executor:  # 원하는 스레드 수로 변경
        for input_file in input_files:
            executor.submit(convert_excel_to_csv, input_file, output_folder)