from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db, AnalysisResult

rating_router = APIRouter()

# 별점 모델
class RatingModel(BaseModel):
    rating: float

@rating_router.post("/{analysis_id}")
async def rate_analysis(analysis_id: int, rating_model: RatingModel, db: Session = Depends(get_db)):
    # 분석 결과 찾기
    analysis_record = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis_record:
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    # 별점 저장
    analysis_record.rating = rating_model.rating
    db.commit()

    return {"message": "Rating submitted.", "rating": rating_model.rating}
