'''
author: @PenguinsResearch © 2022
'''

import re

regexs = {
    r'\d{1,2}/\d{1,2}' : 'dd/mm',
    r'\d{1,2}/\d{4}' : 'mm/yyyy',
    r'\d{1,2}/\d{1,2}/\d{4}': 'dd/mm/yyyy',
    r'ngày \d{1,2}': 'dd',
    r'ngày \d{1,2}/\d{1,2} âm lịch': 'dd/mm',
    r'ngày \d{1,2}/\d{1,2}/\d{4}': 'dd/mm/yyyy',
    r'ngày \d{1,2} tháng \d{1,2} năm \d{4}': 'dd/mm/yyyy',
    r'ngày \d{1,2} tháng \d{1,2} âm lịch': 'dd/mm',
    r'ngày \d{1,2} tháng \d{1,2} , \d{4}': 'dd/mm/yyyy',
    r'ngày \d{1,2} tháng \d{1,2}': 'dd/mm',
    r'ngày \d{1,2} tháng \w+': 'dd/MM',
    r'ngày \d{1,2} tháng \w+ âm lịch': 'dd/MM',
    r'ngày \d{1,2}/\d{1,2}': 'dd/mm',
    r'ngày \d{1,2} tháng \d{1,2} năm \w+ \( tức \d{1,2} tháng \d{1,2} năm \d{4}': 'dd/mm/yyyy',
    r'ngày \d{1,2}.\d{1,2}.\d{4}': 'dd/mm/yyyy',
    r'ngày \d{1,2} tháng \d{1,2} âm lịch': 'dd/mm',
    r'ngày \d{1,2} tháng \d{1,2}, \d{4}': 'dd/mm/yyyy',
    r'ngày mùng \d{1,2}/\d{1,2}': 'dd/mm',
    r'tháng \d{1,2}-\d{4}': 'mm/yyyy',
    r'tháng \d{1,2}': 'mm',
    r'tháng \d{1,2}/\d{4}': 'mm/yyyy',
    r'tháng \d{1,2} âm lịch': 'mm',
    r'tháng \d{1,2} năm \d{4}': 'mm/yyyy',
    r'tháng thứ \d{1,2} của âm lịch \w+': 'mm',
    r'năm \d{4}': 'yyyy',
    r'giữa năm \d{4}': 'yyyy',
    r'\d{4}': 'yyyy',
    r'\d{3}': 'yyyy',
    r'\d{1,2} tháng \d{1,2} năm \d{4}': 'dd/mm/yyyy',
    r'\d{2}/\d{4}': 'mm/yyyy',
    r'\d{1,2}:\d{1,2} utc ngày \d{1,2}/\d{1,2}/\d{4}': 'dd/mm/yyyy',
}


def get_nums(text):
    text = text.replace('/', ' ')
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    text = text.replace(':', ' ')
    text = text.replace('-', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('"', ' ')
    text = text.replace('\'', ' ')
    text = re.sub(' +', ' ', text)
    nums = []
    for word in text.split():
        if word.isdigit():
            nums.append(int(word))
    return nums


def extract_datetime(text):
    matches = []
    text = text.lower()
    for regex, format in regexs.items():
        match = re.search(regex, text)
        if match:
            group = match.group()
            nums = get_nums(group)
            if format == 'dd/mm/yyyy':
                matches.append({
                    'date': nums[0],
                    'month': nums[1],
                    'year': nums[2],
                })
            elif format == 'dd/mm':
                matches.append({
                    'date': nums[0],
                    'month': nums[1],
                    'year': 0,
                })
            elif format == 'mm/yyyy':
                matches.append({
                    'date': 0,
                    'month': nums[0],
                    'year': nums[1],
                })
            elif format == 'mm':
                matches.append({
                    'date': 0,
                    'month': nums[0],
                    'year': 0,
                })
            elif format == 'yyyy':
                matches.append({
                    'date': 0,
                    'month': 0,
                    'year': nums[0],
                })
            elif format == "dd/MM":
                convert_month = {
                    'một': 1,
                    'chạp': 12,
                    'giêng hai': 2,
                    'ba': 3,
                    'tư': 4,
                    'năm': 5,
                    'sáu': 6,
                    'bảy': 7,
                    'tám': 8,
                    'chín': 9,
                    'mười': 10,
                    'tý': 11,
                }
                month = 0
                for key, value in convert_month.items():
                    if key in group:
                        month = value
                        break
                matches.append({
                    'date': nums[0],
                    'month': month,
                    'year': 0,
                })
            else:
                matches.append({
                    'date': 0,
                    'month': 0,
                    'year': 0,
                })
        else:
            matches.append({
                'date': 0,
                'month': 0,
                'year': 0,
            })

    # sort by year, month, date
    matches = sorted(matches, key=lambda x: (
        x['year'], x['month'], x['date']), reverse=True)
    if len(matches) > 0:
        return matches[0]
    return matches


def make_dictionaly(max_range=500_000):
    dictionaly = {}

    basis_number = {
        0 : 'không',
        1 : 'một',
        2 : 'hai',
        3 : 'ba',
        4 : 'tư',
        5 : 'năm',
        6 : 'sáu',
        7 : 'bảy',
        8 : 'tám',
        9 : 'chín',
        10 : 'mười',
    }

    spectrum_number = {
        10 : 'mười',
        100 : 'trăm',
        1000 : 'nghìn',
        1000000 : 'triệu',
        1000000000 : 'tỷ',
    }

    for i in range(0, max_range+1):
        dictionaly[str(i)] = ""
        if i < 11:
            dictionaly[str(i)] = basis_number[i] if i != 4 else "bốn"
        elif i < 20:
            dictionaly[str(i)] = "mười " + basis_number[i%10]
        elif not i%10 and i < 100:
            dictionaly[str(i)] = basis_number[i//10] + " mươi"
        elif i < 100:
            dictionaly[str(i)] = basis_number[i // 10] + " mươi " + basis_number[i % 10]
        elif not i % 100 and i < 1_000:
            dictionaly[str(i)] = basis_number[i // 100] + " trăm"
        elif i < 1_000:
            dictionaly[str(i)] = basis_number[i // 100] + " trăm " + dictionaly[str(i % 100)]
        elif not i % 1_000 and i < 10_000:
            dictionaly[str(i)] = basis_number[i // 1_000] + " nghìn"
        elif i < 10_000:
            dictionaly[str(i)] = dictionaly[str(i // 1_000)] + " nghìn " + dictionaly[str(i % 1000)]
        elif not i % 100_000 and i < 1_000_000:
            dictionaly[str(i)] = basis_number[i // 1_000_000] + " triệu"
        elif i < 1_000_000_000:
            dictionaly[str(i)] = dictionaly[str(i // 1_000_000)] + " triệu " + dictionaly[str(i % 1_000_000)]
        elif not i % 1_000_000_000 and i < 1_000_000_000:
            dictionaly[str(i)] = basis_number[i // 1_000_000_000] + " tỷ"
        elif i < 1_000_000_000:
            dictionaly[str(i)] = dictionaly[str(i // 1_000_000_000)] + " tỷ " + dictionaly[str(i % 1_000_000_000)]
        else:
            dictionaly[str(i)] = "tỉ tỉ"
    
    return dictionaly


_dictionaly = make_dictionaly()
_dictionaly_values = list(_dictionaly.values())[::-1]
_dictionaly_keys = list(_dictionaly.keys())[::-1]
def extract_quantity(text):
    text = text.lower()
    text = text.replace('k', '000')
    text = text.replace('lẻ', ' ')
    text = text.replace('ngàn', 'nghìn')
    text = text.replace('lăm', 'năm')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('(', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace(';', '')
    text = text.replace(':', '')
    text = text.replace('/', ' ')
    text = re.sub(' +', ' ', text)
    text = text.strip().split()

    # replace text to number
    for value in _dictionaly_values:
        value = value.strip().split()
        if all([v in text for v in value]):
            start = text.index(value[0])
            end = start + len(value)
            try:
                text = text[:start] + [_dictionaly_keys[_dictionaly_values.index(' '.join(value))]] + text[end:]
            except: ...
    text = ' '.join(text)
    num = re.sub(r'\D', '', text)
    return num


if __name__ == '__main__':
    # texts = open("datetime.txt", "r", encoding='utf-8').readlines()
    # with open("datetime_output.txt", "w", encoding='utf-8') as f:
    #     for line in texts:
    #         line = line.strip()
    #         f.write(line + '\n')
    #         f.write(str(extract_datetime(line)) + '\n')
    #         f.write('\n')

    text = "Tôi muốn mua năm mươi nghìn lẻ năm cái bánh mì"
    print(extract_quantity(text))
