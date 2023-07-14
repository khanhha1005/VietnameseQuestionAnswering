import os
import re
import json 

wiki_dump = open("wikipedia_20220620_all_wiki.jsonl", "r", encoding="utf-8")

# find all internal links [[wiki/link|label]]
internal_links = re.compile(r'\[\[([^\]]+)\]\]')


def get_and_replace_internal_links(text):
    '''
    [[link|label]] -> label
    return list of internal links and start and end positions,
    replace internal links with plain text
    '''
    links = []
    for t in internal_links.findall(text):
        # find first |
        pipe = t.find('|')
        if pipe < 0:
            link = t
            label = link
        else:
            link = t[:pipe].rstrip()
            label = t[pipe + 1:].strip()

        start = text.find("[[" + t + "]]")
        end = start + len("[[" + t + "]]")
        start_id = text[:start].count(" ") + 1
        end_id = text[:end].count(" ") + 1
        links.append((link, start_id, end_id))
        text = text.replace(t, label)
    
    text = text.replace('[[', '').replace(']]', '')   
    return links, text

def main():
    wiki_dump = open("wikipedia_20220620_all_wiki.jsonl", "r", encoding="utf-8")
    wiki_data = open("wikipedia_20220620_all_wiki_data.jsonl", "w", encoding="utf-8")

    for line in wiki_dump:
        line = json.loads(line)
        id = line["id"]
        revid = line["revid"]
        url = line["url"]
        title = line["title"]
        text = line["text"]

        links, text = get_and_replace_internal_links(text)
        
        data = {
            "id": id,
            "revid": revid,
            "url": url,
            "title": title,
            "text": text,
            "links": links
        }

        wiki_data.write(json.dumps(data) + "\n")
        wiki_data.flush()


def test():

    data = open("wikipedia_20220620_all_wiki.jsonl", "r", encoding="utf-8")
    [data.readline() for i in range(5)] 

    data = json.loads(data.readline())

    text = data["text"]

    links, text = get_and_replace_internal_links(text)

    print(text)

    for link, start, end in links:
        print(link, start, end)

        
if __name__ == "__main__":
    test()
