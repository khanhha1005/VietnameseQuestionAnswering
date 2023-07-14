import json
titles = open("wikipedia_20220620_cleaned/wikipedia_20220620_all_titles.txt", "r", encoding="utf-8")
links = open("wikipedia_20220620_all_links.txt", "r", encoding="utf-8")

titles = [t.strip() for t in titles.readlines() if t.strip() != ""]
links = [l.strip() for l in links.readlines() if l.strip() != ""]

print("Total titles: ", len(titles))
print("Total links: ", len(links))

_links = {}
for line in links:
    line = line.split("\t")
    if len(line) != 3: continue
    label, link, title = line[0], line[1], line[2]
    if not link in _links:
        _links[link] = []
    _links[link].append((label, title))

dict_links = {}
for t in titles:
    _t = "wiki/" + "_".join(t.split(" "))
    if not _t in _links: continue
    _dict = _links[_t]
    for label, title in _dict:
        if label not in dict_links: dict_links[label] = {}
        dict_links[label][title] = _t

for t in titles:
    _t = "wiki/" + "_".join(t.split(" "))
    dict_links[t] = {"wiki" : _t}

print(len(dict_links))

with open("wikipedia_20220620_all_links.json", "w", encoding="utf-8") as f:
    json.dump(dict_links, f, ensure_ascii=False, indent=4)
