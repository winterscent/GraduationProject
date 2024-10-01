from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from consent import consent_router
from file_upload import upload_router
from rating import rating_router

# DB 초기화
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:8080",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(consent_router, prefix="/consent")
app.include_router(upload_router, prefix="/upload")
app.include_router(rating_router, prefix="/rate")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Conversation Analysis Tool"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
