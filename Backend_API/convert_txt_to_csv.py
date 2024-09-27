import csv
import re
import os

pattern = re.compile(r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})(:\d{2})?, (.*) : (.*)')

# txt 파일을 csv로 변환하는 함수
def convert_txt_to_csv(txt_file):
    csv_file = os.path.splitext(txt_file)[0] + '.csv'

    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Date', 'User', 'Message'])

        for line in infile:
            line = line.strip()
            match = pattern.match(line)
            if match:
                year = match.group(1)
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                am_pm = match.group(4)
                hour = match.group(5).zfill(2)
                minute = match.group(6).zfill(2)
                seconds = match.group(7) if match.group(7) else ":00"  # 초가 없으면 00으로 설정
                sender = match.group(8)
                message = match.group(9)

                hour = int(hour)
                if am_pm == '오후' and hour != 12:
                    hour += 12
                elif am_pm == '오전' and hour == 12:
                    hour = 0

                # 최종 DateTime 포맷 설정 (YYYY-MM-DD HH:MM:SS)
                time_24h = f'{hour:02}:{minute:02}{seconds}'
                full_datetime = f"{year}-{month}-{day} {time_24h}"

                csv_writer.writerow([full_datetime, sender, message])

    print(f"'{csv_file}' 파일이 생성되었습니다.")
    return csv_file
