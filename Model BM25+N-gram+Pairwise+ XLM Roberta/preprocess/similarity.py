import os
from copy import deepcopy
import difflib
import time
import sys
import json
import os
from underthesea import text_normalize
import re

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳÝýỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYyYy'


def norm_text(input_str):
    # remove special characters
    input_str = re.sub(f'[^a-zA-Z0-9{s1} ]', '', input_str)
    return input_str

def normalizer(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = norm_text(text)
    text = re.sub(r'\s+', ' ', text)
    return text

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('Time: ', end - start)
        return result
    return wrapper


def n_gram(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


def get_all_candidates(query, sentences):
    """
        :param query: query string
        :param sentences: list of normalized sentences
        :return: list of candidates
    """
    query = text_normalize(query)
    query_tokens = query.lower().split(' ')

    candidates = deepcopy(sentences)
    for i in range(1, len(query_tokens) + 1):
        next_candidates = set()
        for piece in n_gram(query_tokens, i):
            piece_tokens = set(piece.split(' '))
            for sent in candidates:
                if i == 1:
                    if piece in sent.lower():
                        next_candidates.add(sent)
                    # if norm_text(query).lower() == norm_text(sent).lower():
                    #     return [sent]
                else:
                    sent_tokens = set(sent.lower().split(' '))
                    if len(piece_tokens.union(sent_tokens)):
                        next_candidates.add(sent)
        if len(next_candidates) == 0:
            break
        candidates = next_candidates
        # if len(candidates) <= 500:
        #     break
    print('Number of candidates: ', len(candidates))
    return list(candidates)


@timeit
def get_closest_candidates(query, sentences, n=1):
    candidates = get_all_candidates(query, sentences)
    if len(candidates) == 0:
        return None
    return difflib.get_close_matches(query, candidates, n=n, cutoff=0.0)


if __name__ == '__main__':
    query = "ủy Đắk Nông"
    title = "Danh sách Chủ tịch nước Việt Nam"
    top_k = 10
    with open("wikipedia_20220620_all_titles_links.txt", "r", encoding="utf-8") as f:
        sentences = [sentence.lstrip().rstrip() for sentence in f.readlines()]

    links = json.load(open("wikipedia_20220620_all_links.json", "r", encoding="utf-8"))

    candidates = get_closest_candidates(normalizer(query), sentences, n=top_k)

    link = {}
    for candidate in candidates:
        if title in links[candidate]:
            link[candidate] = links[candidate][title]
        elif "wiki" in links[candidate]:
            link[candidate] = links[candidate]["wiki"]
        else:
            last_key = list(links[candidate].keys())[-1]
            link[candidate] = links[candidate][last_key]

    for k, v in link.items():
        print(k, v)
