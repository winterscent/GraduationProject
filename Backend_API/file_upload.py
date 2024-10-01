from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import os
from s3_utils import upload_to_s3
from database import get_db, AnalysisResult
from conversation_analysis import analyze_conversation
from convert_txt_to_csv import convert_txt_to_csv
from name_masking import mask_names_in_csv

upload_router = APIRouter()

@upload_router.post("/")
async def upload_and_analyze_file(
    consent: bool = Form(...),
    file: UploadFile = File(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    db: Session = Depends(get_db)
):
    # 동의 여부 확인
    if not consent:
        return {"message": "Consent not given. Analysis aborted."}

    try:
        # 날짜 유효성 검사
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        if start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Start date must be earlier than end date.")

        # 파일 처리
        file_contents = await file.read()
        file_location = f"files/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as file_object:
            file_object.write(file_contents)

        # 파일 형식에 따라 처리
        if file.filename.endswith('.txt'):
            csv_file = convert_txt_to_csv(file_location)
        elif file.filename.endswith('.csv'):
            csv_file = file_location
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only .txt and .csv are supported.")

        # 이름 마스킹 및 대화 분석
        masked_csv_file = mask_names_in_csv(csv_file)
        conversation_text = extract_conversation_text(masked_csv_file, start_date_obj, end_date_obj)
        analysis_result = analyze_conversation(conversation_text)

        # S3에 파일 업로드 및 DB 저장
        file_url = upload_to_s3(masked_csv_file)
        new_analysis = AnalysisResult(
            file_url=file_url,
            start_date=start_date_obj,
            end_date=end_date_obj,
            result=analysis_result.get("closest_relation")
        )
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)

        # 결과 반환
        return {
            "analysis_id": new_analysis.analysis_id,
            "result": analysis_result.get("closest_relation"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# CSV 파일에서 대화 내용 추출
def extract_conversation_text(masked_csv_file, start_date_obj, end_date_obj):
    conversation_text = []
    with open(masked_csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            row_date = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            if start_date_obj <= row_date <= end_date_obj:
                conversation_text.append(row[2])
    return " ".join(conversation_text)
