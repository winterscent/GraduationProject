from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db, ConsentModel

consent_router = APIRouter()

# 동의 모델
class ConsentForm(BaseModel):
    consent: bool

# 동의 여부 저장
@consent_router.post("/")
async def consent_agreement(consent_form: ConsentForm, db: Session = Depends(get_db)):
    consent_record = ConsentModel(consent=consent_form.consent)
    db.add(consent_record)
    db.commit()
    return {"message": "Consent saved.", "consent": consent_form.consent}

# 동의한 사용자 목록 조회
@consent_router.get("/consented_users/")
async def get_consented_users(db: Session = Depends(get_db)):
    consented_users = db.query(ConsentModel).filter(ConsentModel.consent == True).all()
    return consented_users
