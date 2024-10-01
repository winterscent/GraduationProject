from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db, AnalysisResult

consent_router = APIRouter()

# 동의 모델
class ConsentForm(BaseModel):
    consent: bool

# 동의 여부 저장
@consent_router.post("/")
async def consent_agreement(consent_form: ConsentForm):
    if consent_form.consent:
        return {"message": "Consent accepted.", "consent": True}
    else:
        return {"message": "Consent declined.", "consent": False}
