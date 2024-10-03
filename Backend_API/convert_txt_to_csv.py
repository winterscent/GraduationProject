import csv
import re
import os

# iOS 형식
ios_pattern = re.compile(r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})(:\d{2})?, (.*) : (.*)')

# Android 형식
android_pattern = re.compile(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일 (오전|오후) (\d{1,2}):(\d{2}), (.*) : (.*)')

# Windows 형식
windows_pattern = re.compile(r'\[(.*)\] \[(오전|오후) (\d{1,2}):(\d{2})\] (.*)')


# txt 파일을 csv로 변환하는 함수
def convert_txt_to_csv(txt_file):
    csv_file = os.path.splitext(txt_file)[0] + '.csv'

    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Date', 'User', 'Message'])

        for line in infile:
            line = line.strip()

            # iOS 형식 처리
            match = ios_pattern.match(line)
            if match:
                year = match.group(1)
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                am_pm = match.group(4)
                hour = match.group(5).zfill(2)
                minute = match.group(6).zfill(2)
                seconds = match.group(7) if match.group(7) else ":00"
                sender = match.group(8)
                message = match.group(9)

                hour = int(hour)
                if am_pm == '오후' and hour != 12:
                    hour += 12
                elif am_pm == '오전' and hour == 12:
                    hour = 0

                time_24h = f'{hour:02}:{minute:02}{seconds}'
                full_datetime = f"{year}-{month}-{day} {time_24h}"

                csv_writer.writerow([full_datetime, sender, message])

            # Android 형식 처리
            match = android_pattern.match(line)
            if match:
                year = match.group(1)
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                am_pm = match.group(4)
                hour = match.group(5).zfill(2)
                minute = match.group(6).zfill(2)
                sender = match.group(7)
                message = match.group(8)

                hour = int(hour)
                if am_pm == '오후' and hour != 12:
                    hour += 12
                elif am_pm == '오전' and hour == 12:
                    hour = 0

                time_24h = f'{hour:02}:{minute:02}:00'
                full_datetime = f"{year}-{month}-{day} {time_24h}"

                csv_writer.writerow([full_datetime, sender, message])

            # Windows 형식 처리
            match = windows_pattern.match(line)
            if match:
                sender = match.group(1)
                am_pm = match.group(2)
                hour = match.group(3).zfill(2)
                minute = match.group(4).zfill(2)
                message = match.group(5)

                hour = int(hour)
                if am_pm == '오후' and hour != 12:
                    hour += 12
                elif am_pm == '오전' and hour == 12:
                    hour = 0

                # 여기서 Date 정보가 빠졌으므로, 이전 대화의 날짜를 계속 사용
                if 'full_datetime' in locals():
                    csv_writer.writerow([full_datetime, sender, message])

            # 날짜 정보가 있는 줄 (ex: --- 2023년 8월 1일 화요일 ---) 처리
            date_match = re.match(r'--------------- (\d{4})년 (\d{1,2})월 (\d{1,2})일.* ---------------', line)
            if date_match:
                year = date_match.group(1)
                month = date_match.group(2).zfill(2)
                day = date_match.group(3).zfill(2)
                full_datetime = f"{year}-{month}-{day}"

    print(f"'{csv_file}' 파일이 생성되었습니다.")
    return csv_file
