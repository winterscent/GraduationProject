from fastapi import APIRouter
from pydantic import BaseModel

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
