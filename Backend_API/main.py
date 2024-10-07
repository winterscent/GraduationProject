from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from file_analyze import upload_router
from rating import rating_router

# DB 초기화
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 설정
origins = [
    "http://application-lb-1275420282.ap-northeast-2.elb.amazonaws.com",
    "https://4xsbtut5d7.execute-api.ap-northeast-2.amazonaws.com",
    "https://www.sometime.site",
<<<<<<< HEAD
=======
    "http://103.218.159.57",
    "http://localhost:8080",
>>>>>>> eb836e1 (modified Project)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/upload")
app.include_router(rating_router, prefix="/api/rate")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Conversation Analysis Tool"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
