import csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy as np
import os
import re

# 대화 내용 모델 그룹
conversations = {
    "썸": [
        "이모티콘", "좋아", "좋아해", "웅", "보고 싶어", "뭐해?", "나도", "웃겨", "귀여워", "생각나", "잘자", "같이 가고 싶어", "놀자",
        "궁금해", "보고 싶었어", "배고파", "뭐 먹었어?", "연락해줘", "어디야?", "같이 보고 싶어", "기다려", "마음에 들어", "두근두근",
        "너랑 있으면 좋아", "심쿵", "또 보자", "보고만 있어도 좋아", "같이 걸을까?", "얼굴 보고 싶어", "만나고 싶어", "연락 기다릴게",
        "계속 생각나", "혼자 있을 때 생각나", "나도 보고 싶어", "너무 좋아", "약속 잡자", "다음엔 내가 사줄게", "너랑 얘기하면 편해",
        "잘했어", "자주 연락하자", "만나면 설레", "더 알고 싶어", "가까워지고 싶어", "시간 내줘서 고마워", "지금 생각나", "웃을 때 예뻐",
        "자주 연락해줘", "다음엔 뭐할까?", "점점 좋아져", "서로 알게 돼서 좋다"
    ],

    "연애": [
        "이모티콘", "좋아", "좋아해", "사랑", "사랑해", "웅", "보고 싶어", "뭐해?", "나도", "보고 싶었어", "같이 있고 싶어", "너밖에 없어",
        "고마워", "진심이야", "행복해", "잘자", "내 생각해?", "내가 더 사랑해", "내 사람", "너를 지킬게", "우리", "평생 함께 하자",
        "나도 너 사랑해", "너무 보고 싶어", "오늘 너무 좋았어", "넌 나의 전부야", "매일 보고 싶어", "항상 생각해", "널 위해 뭐든지 할게",
        "같이 늙어가고 싶어", "미래를 같이 그리고 싶어", "영원히 사랑해", "넌 나의 행복이야", "너 없으면 안 돼", "함께라서 행복해", "오늘도 예뻐",
        "같이 가자", "항상 고마워", "언제나 네 편이야", "넌 나의 운명이야", "같이 살자", "계속 너만 생각나", "널 위해서라면 뭐든 할 수 있어",
        "함께 있으면 편해", "함께 여행 가자", "너만 믿어", "영원히 함께하자", "같이 있어줘서 고마워", "나에게는 너뿐이야", "마음이 따뜻해져",
        "결혼하자", "서로 의지하자", "서로를 지켜주자", "네가 있어서 행복해", "내 사랑", "매일 아침 일어나면 네 생각이 나", "너와 함께라면 뭐든 좋아"
    ],

    "친구": [
        "야", "아니", "너", "ㅇㅇ", "ㅇㅋ", "ㄴㄴ", "ㅇㅈ", "ㄷㄷ", "개", "꺼져", "헐", "오바야", "대박", "와우", "그래", "ㅋㅋ",
        "장난해?", "쩔어", "뭐래?", "진짜?", "인정", "놀자", "말도 안 돼", "어쩔", "킹받네", "좀 과한데?", "개웃겨", "쩐다", "ㅇㅉ?",
        "뭐하냐?", "미쳤냐?", "놀러가자", "콜", "노잼", "개좋아", "시간 나면 불러", "찐친", "배고프다", "담에 보자", "어디서 볼래?",
        "가자", "대환장 파티", "완전 실화냐?", "뭐하냐", "그만해", "계속 말해", "우와", "진지하게", "뜬금없다", "오케이", "그냥 그래",
        "솔직히", "개쩐다", "솔직히 말해", "그래? 난 아닌데", "솔직히 재밌어", "어이없네", "아니 ㅋㅋ", "진짜?", "ㅋㅋㅋ", "개노잼", "에반데",
        "개에반데", "지랄", "ㅈㄹ", "염병", "개소리", "ㅅㅂ", "시발", "씨발", "ㅂㅅ", "병신", "등신", "ㅈㄴ", "존나", "겁나", "야랄",
        "좆", "ㅈ같네", "ㅈ같다", "좃같네", "좃같다", "좆같네", "좆같다"
    ],

    "비즈니스": [
        "알겠습니다", "확인했습니다", "감사합니다", "그때 뵙겠습니다", "드립니다", "바랍니다", "네", "회신 부탁드립니다",
        "진행 상황을 공유드립니다", "협조 부탁드립니다", "첨부 파일 확인 부탁드립니다", "논의하고 싶습니다", "일정 조율 부탁드립니다",
        "미팅 가능합니다", "검토 후 피드백 주시면 감사하겠습니다", "감사합니다", "수고하셨습니다", "최종 확인 부탁드립니다",
        "의견 주시면 감사하겠습니다", "추가 자료 요청드립니다", "다시 확인 부탁드립니다", "프로젝트 진행 상황 보고드립니다", "목표 달성하겠습니다",
        "앞으로 잘 부탁드립니다", "예산 검토 요청드립니다", "계약서 확인 부탁드립니다", "견적 요청드립니다", "빠른 답변 부탁드립니다",
        "팀과 논의 후 연락드리겠습니다", "문제 해결 방안을 찾고 있습니다", "서비스 향상을 위해 노력하겠습니다", "의견 주셔서 감사합니다",
        "수고 많으셨습니다", "업무 진행에 차질 없도록 하겠습니다", "피드백 감사합니다", "고객 만족을 위해 최선을 다하겠습니다", "이슈가 발생했습니다",
        "지원 요청드립니다", "검토 부탁드립니다", "좋은 결과 기대하겠습니다", "회의 시간 조정 부탁드립니다", "진행 상태 업데이트 부탁드립니다",
        "예산 관련 문의드립니다", "상의 후 회신 부탁드립니다", "감사합니다", "서류 작성 부탁드립니다", "자료 준비 중입니다",
        "추가 정보 요청드립니다", "더 좋은 방향으로 진행하겠습니다", "감사드리며 지속적인 협조 부탁드립니다", "곧 회신 드리겠습니다",
        "계획서 전달드립니다", "담당자와 상의하겠습니다", "추가 의견 부탁드립니다"
    ]
}

# 대화 내용 모델 그룹을 하나의 문서로 합침
combined_conversations = {relation: ' '.join(conversation) for relation, conversation in conversations.items()}

# 형태소 분석기 초기화(Okt 사용)
translate = Okt()

# 대화 내용 모델 그룹을 형태소 분석
combined_tokens = {relation: translate.morphs(text) for relation, text in combined_conversations.items()}
combined_for_vectorize = {relation: ' '.join(tokens) for relation, tokens in combined_tokens.items()}

# TF-IDF 벡터 생성
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(list(combined_for_vectorize.values()))


# 대화 내용과 관계 유형 간의 유사도 계산
def classify_relationship(target_input):
    target_tokens = translate.morphs(target_input)
    target_for_vectorize = ' '.join(target_tokens)
    t_vec = vectorizer.transform([target_for_vectorize])  # 대상 문장의 TF-IDF 벡터 생성

    # 코사인 유사도 계산
    cosine_similarities = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector])
        cosine_similarities[relation] = cosine_similarity(t_vec, combined_vec)[0][0]

    # 유클리드 거리 계산
    euclidean_distances = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector]).toarray()[0]
        target_vec_arr = t_vec.toarray()[0]
        euclidean_distances[relation] = euclidean(combined_vec, target_vec_arr)

    # 최종 점수 계산 (코사인 유사도는 최대화, 유클리드 거리는 최소화가 목적)
    final_scores = {}
    max_euclidean = np.max(list(euclidean_distances.values()))
    for relation in cosine_similarities.keys():
        cosine_score = cosine_similarities[relation]
        euclidean_score = euclidean_distances[relation]
        adjusted_euclidean_score = 1 - (euclidean_score / max_euclidean)
        final_score = (cosine_score + adjusted_euclidean_score) / 2
        final_scores[relation] = final_score

    closest_relation_final = max(final_scores, key=final_scores.get)

    return cosine_similarities, euclidean_distances, final_scores, closest_relation_final


def read_csv_file_by_date(filename, start_date, end_date, column_index):
    conversation = ""
    first_date = None
    last_date = None
    date_format = "%Y-%m-%d %H:%M:%S"  # CSV 파일의 날짜 형식 정의

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵

        for row in reader:
            try:
                row_date = datetime.strptime(row[0], date_format).strftime("%Y%m%d")  # 날짜 형식을 YYYYMMDD로 통일
                if not first_date:
                    first_date = row_date  # 첫 번째 날짜 저장
                last_date = row_date  # 마지막 날짜 업데이트
            except ValueError:
                print("잘못된 날짜 형식이 포함된 행을 건너뜁니다.")
                continue
            if start_date <= row_date <= end_date:
                conversation += row[column_index] + " "

    return conversation, first_date, last_date


def read_txt_file_by_date(filename, start_date, end_date):
    conversation = ""
    first_date = None
    last_date = None

    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

        # 날짜 및 시간 패턴을 추출하기 위한 정규식
        pattern = r"(\d{4}\. \d{1,2}\. \d{1,2}\. .*\d{1,2}:\d{2}), (.*?): (.*)"
        matches = re.findall(pattern, content)

        for match in matches:
            date_str, user, message = match
            try:
                row_date = datetime.strptime(date_str, "%Y. %m. %d. %p %I:%M").strftime(
                    "%Y%m%d")  # 예: "2022. 6. 8. 오후 7:46"
                if not first_date:
                    first_date = row_date  # 첫 번째 날짜 저장
                last_date = row_date  # 마지막 날짜 업데이트
            except ValueError:
                print("잘못된 날짜 형식을 건너뜁니다.")
                continue
            if start_date <= row_date <= end_date:
                conversation += message + " "

    return conversation, first_date, last_date


def validate_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y%m%d")
    except ValueError:
        return None


def get_valid_file_path():
    while True:
        filename = input("CSV 또는 TXT 파일 경로를 입력하세요 (예: filename.csv 또는 filename.txt): ")
        if os.path.exists(filename) and (filename.endswith('.csv') or filename.endswith('.txt')):
            return filename
        else:
            print("파일이 없거나 잘못된 형식입니다. 다시 입력해주세요.")


# 파일 경로 입력받기
filename = get_valid_file_path()

# 파일 형식에 따라 첫 번째와 마지막 날짜 추출 및 대화 내용 추출
if filename.endswith('.csv'):
    column_index = 2  # 분석할 열의 인덱스 (대화 내용이 있는 열)
    start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
    start_date = validate_date(start_date_str)
    while start_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
        start_date = validate_date(start_date_str)

    end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
    end_date = validate_date(end_date_str)
    while end_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
        end_date = validate_date(end_date_str)

    # 대화 내용을 읽어오기
    conversations, first_date, last_date = read_csv_file_by_date(filename, start_date, end_date, column_index)

else:  # TXT 파일 처리
    start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
    start_date = validate_date(start_date_str)
    while start_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
        start_date = validate_date(start_date_str)

    end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
    end_date = validate_date(end_date_str)
    while end_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
        end_date = validate_date(end_date_str)

    # 대화 내용을 읽어오기
    conversations, first_date, last_date = read_txt_file_by_date(filename, start_date, end_date)

# 파일에 포함된 날짜 출력
print(f"파일에 포함된 날짜: {first_date} ~ {last_date}")

# 대화 내용과 관계 유형 판단
cosine_similarities, euclidean_distances, final_scores, closest_relation_final = classify_relationship(conversations)

print("코사인 유사도 기반 점수(1에 가까울수록 좋음):")
for relation, score in cosine_similarities.items():
    print("{}: {:.4f}".format(relation, score))

print("\n유클리드 거리 기반 점수(0에 가까울수록 좋음):")
for relation, score in euclidean_distances.items():
    print("{}: {:.4f}".format(relation, score))

print("\n최종 점수(1에 가까울수록 좋음):")
for relation, score in final_scores.items():
    print("{}: {:.4f}".format(relation, score))

print("\n입력한 대화 내용은 '{}' 관계에 해당합니다.".format(closest_relation_final))
