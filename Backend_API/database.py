from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB 설정 (AWS EC2 환경에서 DB URL을 입력)
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:hellomac@mariadb.cr0662wm0o8o.ap-northeast-2.rds.amazonaws.com:3306/analysis"

# DB 엔진 및 세션 초기화
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 분석 결과 테이블 정의
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    analysis_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    file_url = Column(String(255), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    result = Column(String(8), nullable=True)
    score = Column(Integer, nullable=True)
    comment = Column(String(255), nullable=True)

# DB 세션 획득 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
