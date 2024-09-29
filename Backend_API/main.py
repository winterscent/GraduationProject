from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import os
import csv
from convert_txt_to_csv import convert_txt_to_csv
from name_masking import mask_names_in_csv
from conversation_analysis import analyze_conversation
from datetime import datetime

app = FastAPI()


# 기본 페이지 설정
@app.get("/")
async def read_root():
    return {"message": "Welcome to the File Upload API. Use the /uploadfile/ endpoint to upload .txt or .csv files."}


# 업로드된 파일을 처리하고 분석하는 엔드포인트
@app.post("/uploadfile/")
async def upload_file(
        file: UploadFile = File(...),
        start_date: str = Form(...),
        end_date: str = Form(...)
):
    try:
        # 날짜 유효성 검사
        start_date_obj = datetime.strptime(start_date, "%Y%m%d")
        end_date_obj = datetime.strptime(end_date, "%Y%m%d")
        if start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Start date must be earlier than end date.")

        # 파일 내용 메모리에 읽기
        file_contents = await file.read()  # 비동기로 파일 내용 읽기
        file_location = f"files/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb") as file_object:
            file_object.write(file_contents)  # 파일 내용 저장

        # 파일이 .txt라면 .csv로 변환
        if file.filename.endswith('.txt'):
            csv_file = convert_txt_to_csv(file_location)
        elif file.filename.endswith('.csv'):
            csv_file = file_location
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only .txt and .csv are supported.")

        # CSV 파일에서 이름 마스킹
        masked_csv_file = mask_names_in_csv(csv_file)

        # 마스킹된 CSV 파일에서 대화 내용 분석
        with open(masked_csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)  # 헤더 읽기
            conversation_text = []

            # 날짜 범위에 맞는 대화 내용 필터링
            for row in csv_reader:
                row_date = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")  # 1번째 열에서 날짜 확인
                if start_date_obj <= row_date <= end_date_obj:
                    conversation_text.append(row[2])  # 3번째 열에서 대화 내용 추출

        # 대화 내용 분석
        analysis_result = analyze_conversation(" ".join(conversation_text))

        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
