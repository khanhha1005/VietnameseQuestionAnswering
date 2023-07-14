'''
author: @PenguinsResearch © 2022
'''

import pandas as pd


def is_query_datatime(question, answer):
    _datetime = ['ngày', 'tháng', 'năm']
    for word in _datetime:
        if word in answer:
            return True
    _question = [
        'năm nào',
        'tháng nào',
        'ngày nào',
        'năm bao nhiêu',
        'tháng bao nhiêu',
        'ngày bao nhiêu',
        'khi nào',
        'lúc nào',
        'thời gian nào',
        'ngày mấy',
    ]
    for word in _question:
        if word in question:
            return True
    return False


def is_query_quantity(question, answer):
    def is_num(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    if is_num(answer):
        return True

    _question = [
        'bao nhiêu',
        'bao lâu',
        'bao giờ',
    ]

    for word in _question:
        if word in question:
            return True


CATEGORY = {
    'datetime': is_query_datatime,
    'quantity': is_query_quantity,
    'wiki': lambda question, answer: 'wiki' in answer,
}


def get_category(question, answer, short_candidate):
    for category, func in CATEGORY.items():
        if func(question, answer):
            return category
    return 'other'


def get_category_from_df(df):
    df['category'] = df.apply(lambda row: get_category(
        row['question'], row['answer'], row['short_candidate']), axis=1)
    return df


def get_category_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return get_category_from_df(df)


if __name__ == "__main__":
    import json
    data = json.load(open(
        '/code/saved_models/data/zac2022/zac2022_context_question.json', 'r', encoding='utf-8'))
    df = pd.DataFrame(columns=['question', 'answer', 'short_candidate'])
    for item in data:
        df = df.append({
            'question': item['question'],
            'answer': item['answer'],
            'short_candidate': item['short_candidate'],
        }, ignore_index=True)

    df = get_category_from_df(df)
    # df = get_category_from_csv("zac2022_context_question.csv")
    print(df['category'].value_counts())
    df.to_csv("/code/zac2022_context_question_category.csv", index=False)
