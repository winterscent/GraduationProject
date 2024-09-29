from sqlalchemy import create_engine, Column, Integer, Boolean, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 동의 모델
class ConsentModel(Base):
    __tablename__ = "consents"
    id = Column(Integer, primary_key=True, index=True)
    consent = Column(Boolean, default=False)

# 분석 결과 모델
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    start_date = Column(String)
    end_date = Column(String)
    masked_csv_file = Column(String)
    analysis_result = Column(String)
    rating = Column(Float, nullable=True)

# DB 연결 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
