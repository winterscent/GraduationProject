from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import csv
import shutil
from convert_txt_to_csv import convert_txt_to_csv
from name_masking import mask_names_in_csv
from conversation_analysis import analyze_conversation

app = FastAPI()


# 기본 페이지 설정
@app.get("/")
async def read_root():
    return {"message": "Welcome to the File Upload API. Use the /uploadfile/ endpoint to upload .txt or .csv files."}


# 업로드된 파일을 처리하고 분석하는 엔드포인트
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 파일을 서버에 저장
        file_location = f"files/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        # 파일이 .txt라면 CSV로 변환
        if file.filename.endswith('.txt'):
            csv_file = convert_txt_to_csv(file_location)
        elif file.filename.endswith('.csv'):
            csv_file = file_location
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only .txt and .csv are supported.")

        # CSV 파일에서 이름을 마스킹
        masked_csv_file = mask_names_in_csv(csv_file)

        # 마스킹된 CSV 파일에서 대화 내용을 분석 (대화 내용이 3번째 열에 있다고 가정)
        with open(masked_csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # 헤더 스킵
            conversation_text = " ".join([row[2] for row in csv_reader])  # 3번째 열에서 대화 내용 추출

        # 대화 내용을 분석
        analysis_result = analyze_conversation(conversation_text)

        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
