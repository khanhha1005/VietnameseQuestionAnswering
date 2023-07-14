import json

data_dump = open("wikipedia_20220620_all_wiki.jsonl", "r", encoding="utf-8")
titles = open("wikipedia_20220620_all_titles.txt", "w", encoding="utf-8")


for line in data_dump:
    data = json.loads(line)
    title = data["title"]
    titles.write(title + "\n")
    titles.flush()
