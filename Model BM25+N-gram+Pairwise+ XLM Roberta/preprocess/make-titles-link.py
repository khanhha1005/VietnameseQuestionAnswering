import os
import json

links = json.load(open('wikipedia_20220620_all_links.json', 'r', encoding='utf-8'))

with open('wikipedia_20220620_all_titles_links.txt', 'w', encoding='utf-8') as f:
    for k, v in links.items():
        f.write(k + '\n')
