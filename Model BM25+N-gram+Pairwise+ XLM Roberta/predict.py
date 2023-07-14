'''
author: @PenguinsResearch © 2022
'''

# ignore warning "UserWarning: You seem to be using the pipelines sequentially..."
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import os
import re
import time
import json
import dill
import difflib
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.utils import remove_accents, norm_text
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from model.reader.base import Question
from model.retriever.pyserini_retriever import retriever, build_searcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from underthesea import text_normalize
from model.utils.utils_entity import extract_datetime, extract_quantity
from model.utils import s1


# load config from config file
with open('/code/config.json') as f:
    args = json.load(f)


class Args:
    def __init__(self, args):
        self.__dict__.update(args)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


args = Args(args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time_load_model = time.time()

# Retriever
args.index_path = "/code/saved_models/indexes/wikipedia_20220620"
searcher = build_searcher(args)
# Reader
# model_name_or_path = "/code/saved_models/vi-mrc-base"
model_name_or_path = "/code/saved_models/vi-mrc-large"
tokenizer_qa = AutoTokenizer.from_pretrained(model_name_or_path)
model_qa = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
nlp = pipeline('question-answering', model=model_qa,
               tokenizer=tokenizer_qa, device=0 if torch.cuda.is_available() else -1)
# SentenceTransformer
model_st = SentenceTransformer(
    '/code/saved_models/vn_sbert/phobert_base_mean_tokens_NLI_STS').eval().to(device)
# Classifier
model_cls = dill.load(
    open('/code/saved_models/classifier/classifier_model.sav', 'rb'))

zac2022_titles = open(
    "/code/saved_models/data/wikipedia_20220620_all_titles_links.txt", 'r', encoding='utf-8').readlines()
zac2022_titles = [title.strip() for title in zac2022_titles
                  if not "(định hướng)" in title and not "(Định hướng)" in title]
zac2022_links = json.load(
    open("/code/saved_models/data/wikipedia_20220620_all_links.json", "r", encoding="utf-8"))

end_time_load_model = time.time()
print("Load model time: ", end_time_load_model - start_time_load_model)

def normalizer(text):
    text = re.sub(r'\([^)]*\)', '', text)

    text = norm_text(text)
    text = re.sub(r'\s+', ' ', text)
    return text

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

    candidates = sentences[:]
    for i in range(1, len(query_tokens) + 1):
        next_candidates = set()
        for piece in n_gram(query_tokens, i):
            piece_tokens = set(piece.split(' '))
            for sent in candidates:
                if i == 1:
                    if piece in sent.lower():
                        next_candidates.add(sent)
                else:
                    sent_tokens = set(sent.lower().split(' '))
                    if len(piece_tokens.union(sent_tokens)):
                        next_candidates.add(sent)
        if len(next_candidates) == 0:
            break
        candidates = next_candidates
    return list(candidates)


def get_closest_candidates(query, sentences, n=1):
    candidates = get_all_candidates(query, sentences)
    if len(candidates) == 0:
        return None
    return difflib.get_close_matches(query, candidates, n=n, cutoff=0.0)


def remove_dublicated(text):
    text = text.split(' ')
    substring = [text[i]
                 for i in range(len(text)) if text[i] not in text[i+1:]]
    return ' '.join(substring)


def custom_matches(a, b, ngram):
    a_pieces = n_gram(a, ngram)
    b_pieces = n_gram(b, ngram)
    count = 0
    for piece in a_pieces:
        if piece in b_pieces:
            count += 1
    return count / np.mean([len(a_pieces), len(b_pieces)])


def get_link(candidates, zac2022_links, title="wiki"):
    link = {}
    for candidate in candidates:
        if title in zac2022_links[candidate]:
            link[candidate] = zac2022_links[candidate][title]
        elif "wiki" in zac2022_links[candidate]:
            link[candidate] = zac2022_links[candidate]["wiki"]
        else:
            last_key = list(zac2022_links[candidate].keys())[0]
            link[candidate] = zac2022_links[candidate][last_key]
    return link


def get_majority_vote(candidates, weights=[4, 2, 2, 2, 2, 1, 1, 1, 1, 1], similar_threshold=0.5, ngram=None):
    if weights is None:
        weights = np.ones(len(candidates))

    def norm(input_str):
        # remove special characters
        input_str = re.sub(f'[^a-zA-Z0-9{s1} ]', '', input_str)
        return input_str

    # assert len(candidates) == len(weights), 'candidates and weights must have the same length'
    if len(candidates) != len(weights):
        return None

    candidate_lengths = [len(c.split(' ')) for c in candidates]
    normed_candidates = [norm(c.lower()) for c in candidates]
    normed_candidates = [c.replace("bullet", "") for c in normed_candidates]
    normed_candidates = [(i, c0, c, w) for i, (c0, c, w) in enumerate(
        zip(candidates, normed_candidates, weights)) if len(c) > 0]
    indexes, candidates, normed_candidates, weights = zip(*normed_candidates)

    if ngram is None:
        ngram = np.ceil(np.median(candidate_lengths)).astype(int)
    if np.mean(candidate_lengths) < 2:
        ngram = 1

    # consider the first one
    if normed_candidates[0] in normed_candidates[1:]:
        _candidates = []
        for i, n in enumerate(normed_candidates[1:]):
            if normed_candidates[0] in n:
                _candidates.append((i+1, n))
        avg_len = np.ceil(np.mean([len(c.split(' ')) for _, c in _candidates]))
        for i, c in _candidates:
            if len(c.split(' ')) > avg_len:
                return (i, candidates[0])
        return (0, candidates[0])

    position = {normed_candidates[0]: [0]}
    count_votes = {normed_candidates[0]: weights[0]}
    for i in range(1, len(normed_candidates)):
        flag = False
        for k in count_votes.keys():
            if custom_matches(normed_candidates[i], k, ngram) >= similar_threshold:
                count_votes[k] += weights[i]
                flag = True
                position[k].append(i)
                break
        if not flag:
            count_votes[normed_candidates[i]] = weights[i]
            position[normed_candidates[i]] = [i]

    best_normed_candidate = sorted(
        count_votes.keys(), key=lambda x: count_votes[x], reverse=True)[0]
    candidates = [candidates[i] for i in position[best_normed_candidate]]
    indexes = [indexes[i] for i in position[best_normed_candidate]]
    candidate_lengths = [len(c.split(' ')) for c in candidates]
    avg_len = np.ceil(np.mean(candidate_lengths))
    for i, c in zip(indexes, candidates):
        if len(c.split(' ')) == avg_len:
            return (i, c)
    return (0, candidates[0])


top_k = 10
w_read = 0.6
w_rank = 0.2
w_sim = 0.2

# threshold = 0.7
# sim_threshold = 0.5
# read_threshold = 0.35
# rank_threshold = 0.5
# dist_threshold = 0.5

threshold = 0.1
sim_threshold = 0.1
read_threshold = 0.1
rank_threshold = 0.1
dist_threshold = 0.1


def get_answer(question, category='wiki'):
    if question[-1] != "?":
        question += " ?"  # add question mark if not exist

    question = Question(question, language='vi')

    # Retriever
    contexts = retriever(question, searcher, top_k)

    # Reader
    results = []
    _contexts = []

    # norm c.score to [0, 1]
    max_score = max([c.score for c in contexts])
    min_score = min([c.score for c in contexts])
    for c in contexts:
        c.score = (c.score - min_score) / (max_score - min_score)

    for rank, c in enumerate(contexts):
        lines = c.text.splitlines()
        rank_score = c.score
        for line in lines:
            context = line.strip()
            if not len(context):
                continue
            _contexts.append([c.title, rank_score, context])

    for title, rank_score, _context in _contexts:
        QA_input = {
            'question': question.text,
            'context': _context
        }

        res = nlp(QA_input)
        wiki_link = None
        results.append([title,
                        remove_dublicated(res['answer']),
                        res['score'],
                        res['score'], rank_score,
                        _context,
                        wiki_link,
                        res['start'],
                        res['end'],
                        None,
                        ])

    query = question.text
    if category == "wiki":
        sents = ["_".join(r[0].split("_")[:-1]) + " . " + r[5]
                 for r in results]
    else:
        sents = [r[5] for r in results]
    sents_embeddings = model_st.encode(
        [query] + sents, show_progress_bar=False)
    query_embedding = sents_embeddings[0]
    sents_embeddings = sents_embeddings[1:]

    dists = pairwise_distances(query_embedding.reshape(
        1, -1), sents_embeddings, metric='cosine')[0]
    dists = 1 - dists

    # norm dists to [0, 1]
    dists = (dists - dists.min()) / (dists.max() - dists.min())

    # norm read_score to [0, 1]
    read_scores_max = max([r[3] for r in results])
    read_scores_min = min([r[3] for r in results])

    for i, r in enumerate(results):
        distance = dists[i]
        if distance < sim_threshold:
            distance = 0
        r.append(distance)

        results[i][3] = (r[3] - read_scores_min) / \
            (read_scores_max - read_scores_min)

    for i in range(len(results)):
        if results[i][3] < read_threshold:
            results[i][3] = 0

        if results[i][4] < rank_threshold:
            results[i][4] = 0

    results = [[t, a, s * w_read + rs * w_rank + d * w_sim, as_, rs, c, wiki_link, d, start, end, candidates]
               for t, a, s, as_, rs, c, wiki_link, start, end, candidates, d in results]

    if len(results) == 0:
        results.append(["null", 0, 0, 0, "null"])

    # norm score to [0, 1]
    max_score = max([r[2] for r in results])
    min_score = min([r[2] for r in results])
    for r in results:
        r[2] = (r[2] - min_score) / (max_score - min_score)

        if r[2] < threshold:
            r[2] = 0

    df = pd.DataFrame(results, columns=['title',
                      'answer', 'score', 'reader_score',
                                        'rank_score', 'context',
                                        'wiki_link', 'dists', 'start', 'end',
                                        'candidates',
                                        ])

    # df = df[df['score'] > 0].reset_index(drop=True)
    df = df.sort_values(by=['score'], ascending=False)

    # rerank
    df['reranked'] = [0] * len(df)
    # if category == "wiki":
    #     candidates = df['answer'].tolist()
    #     index, candidate = get_majority_vote(candidates, ngram=1)
    #     if index > 0:
    #         df = df.iloc[[
    #             index] + [i for i in range(len(df)) if i != index]].reset_index(drop=True)
    #         df.loc[0, 'reranked'] = index

    # reorder columns
    df = df[['reranked', 'title', 'answer', 'candidates', 'score', 'reader_score',
             'rank_score', 'dists', 'start', 'end', 'wiki_link', 'context']]
    df.index.name = 'rank'
    df = df.reset_index()
    if category == "wiki":
        candidates = get_closest_candidates(
            normalizer(df.iloc[0]['answer']), zac2022_titles, n=1)
        candidate_links = get_link(candidates, zac2022_links, "_".join(
            df.iloc[0]['title'].split("_")[:-1]))

        df.loc[0, 'candidates'] = candidate_links[candidates[0]]
    return df


def formate_answer(answers, top_k=10, category='wiki'):
    results = []
    _results = []
    for i, row in answers[:top_k].iterrows():
        answer = row['answer']
        candidates = row['candidates']
        score = row['score']
        reader_score = row['reader_score']
        rank_score = row['rank_score']
        context = row['context']

        if answer == "null":
            results.append(None)

        if category == 'datetime':
            answer_entity = extract_datetime(answer)
            date = answer_entity['date']
            month = answer_entity['month']
            year = answer_entity['year']
            _answer = ""
            if date:
                _answer += f"ngày {date}"
            if month:
                _answer += f" tháng {month}"
            if year:
                _answer += f" năm {year}"
            _answer = _answer.strip()
            if _answer:
                results.append(_answer)

        elif category == 'quantity':
            ''' quantity '''
            answer = extract_quantity(answer)
            if len(answer) == 0:
                answer = None
            results.append(answer)

        elif category == 'wiki':
            if candidates:
                results.append(candidates)
            else:
                results.append(None)
        else:
            results.append(None)

    _results = sorted(_results, key=lambda x: x[1] + x[2] + x[3], reverse=True)
    for r in _results:
        results.append(r[0])

    results = list(dict.fromkeys(results))  # remove duplicate and keep order
    return {
        "answers": results,
        "raw_answers": answers["answer"].tolist(),
    }


def init(question="Bác Hồ sinh ngày tháng năm nào?"):
    # Classifier
    category = model_cls.predict(question)[0]
    df = get_answer(question, category)
    ans = formate_answer(df, top_k=top_k, category=category)
    return ans['answers'][0] if len(ans['answers']) > 0 else None


def predict(question):
    # Classifier
    category = model_cls.predict(question)[0]
    df = get_answer(question, category)
    ans = formate_answer(df, top_k=top_k, category=category)
    return ans['answers'][0] if len(ans['answers']) > 0 else None


def main():
    init()

    save_dir = "/result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_questions = json.load(
        open("/data/zac2022_testb_only_question.json", "r", encoding="utf-8"))

    total_question = data_questions['_count_']
    test_cases = data_questions['data']
    all_result = []

    start_time_predict = time.time()

    for question in tqdm(test_cases, total=total_question, desc="Predicting"):
        question_id = question['id']
        question = question['question']
        answer = predict(question)
        all_result.append({
            "id": question_id,
            "question": question,
            "answer": answer
        })

    end_time_predict = time.time()
    print("Total time predict: ", end_time_predict - start_time_predict)

    submission = {"data": all_result}
    with open(os.path.join(save_dir, "submission.json"), "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
