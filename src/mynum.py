#import pyocr
#import pyocr.builders
#import re
#import datetime
#from PIL import Image
#import easyocr
#import cv2
#import numpy as np


#def detect_gender(txt_gender):
    # 性別を検索する文字列
    #gender_match = "性別"

    #for result in txt_gender:
        # テキストと確信度の取得
        #text = result[1]
        #confidence = result[2]

        # "性別"がテキストに含まれる場合
        #if gender_match in text:
            # "性別"の直後にあるテキストを取得
            #index = text.find(gender_match)
            #gender = text[index+len(gender_match):].strip()

            #if '男' in gender:
                #return 'male'
            #elif '女' in gender:
                #return 'female'
            #else:
                #return 'unknown'
            


# 生年月日の抽出
#def detect_birthdate(txt_age):
    # 正規表現パターンの定義
    #birthdate_regex = r'(平成|昭和|大正|明治|令和)(\d+|[元一二三四五六七八九十]+)年\s+(\d+月\d+日)生'
    # 正規表現にマッチする部分を検索
    #birthdate_match = re.search(birthdate_regex, txt_age)
    # マッチした部分があれば生年月日を取得
    #if birthdate_match:
        #era = birthdate_match.group(1)
        #year = birthdate_match.group(2)
        #if year == "元":
            #year = "1"

        #if era == "平成":
            #year_number = int(year) + 1988
            #date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        #elif era == "昭和":
            #year_number = int(year) + 1925
            #date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        #elif era == "大正":
            #year_number = int(year) + 1911
            #date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        #elif era == "明治":
            #year_number = int(year) + 1867
            #date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        #elif era == "令和":
            #year_number = int(year) + 2018
            #date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        #else:
            #date = era + year + '年' + birthdate_match.group(3)  # 元号 + 年数 + 年月日

        # 生年月日をdatetime型に変換
        #birth_day = datetime.datetime.strptime(date, '%Y年%m月%d日')

        # 現在の日付を取得
        #current_date = datetime.datetime.now()

        # 年齢を計算
        #age = current_date.year - birth_day.year - ((current_date.month, current_date.day) < (birth_day.month, birth_day.day))
        
        #return age
    #else:
        #return None




