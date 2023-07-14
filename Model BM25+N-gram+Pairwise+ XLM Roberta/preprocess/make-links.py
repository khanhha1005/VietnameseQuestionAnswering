import os
import re
import json 
from tqdm import tqdm

docs_data = open("wikipedia_20220620_all_wiki.jsonl", "r", encoding="utf-8")
titles_data = open("wikipedia_20220620_cleaned/wikipedia_20220620_all_titles.txt", "r", encoding="utf-8").readlines()
titles_data = ["wiki/" + "_".join(title.strip().split(" ")) for title in titles_data]

# find all internal links [[wiki/link|label]]
internal_links = re.compile(r'\[\[([^\]]+)\]\]')

def get_and_replace_internal_links(text):
    '''
    [[link|label]] -> label
    return list of internal links and start and end positions,
    replace internal links with plain text
    '''
    links = []
    labels = []
    for t in internal_links.findall(text):
        # find first |
        pipe = t.find('|')
        if pipe > 0:
            link = t[:pipe].rstrip()
            label = t[pipe + 1:].strip()
            if not link.startswith('wiki/'): continue
            links.append(link)
            labels.append(label)

    return links, labels

with open("wikipedia_20220620_all_links.txt", "w", encoding="utf-8") as f:
    for doc in tqdm(docs_data, total=len(titles_data), desc="Processing docs"):
        doc = json.loads(doc)
        title = doc['title']
        links, labels = get_and_replace_internal_links(doc["text"])
        for link, label in zip(links, labels):
            f.write(f"{label}\t{link}\t{title}\n")
        f.flush()
