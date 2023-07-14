import json 
import re 
import os 

from tqdm import tqdm 

titles = open('wikipedia_20220620_all_titles.txt').read().splitlines()
titles = [ t.strip() for t in titles ]

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

def clean(text):
    # clean BULLET::
    text = re.sub(r'BULLET\s*::::\s*', '', text)

    text = text.replace('\n', ' ').replace('\r', ' ')
    return text 


class WikiDoc:
    IGNORES_SECTIONS = [
        "Xem thêm.",
        "Tham khảo.",
        "Liên kết ngoài.",
        "Ngoài lề.",
    ]

    def __init__(self, title:str, source:str):
        self.source = source
        self.title = title

    def get_sections(self):
        sections = []
        rank = -1
        for line in self.source.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith("="):
                rank = line.count("=") // 2
                title = line.replace("=", "").strip()
                if rank == 2:
                    sections.append({
                        "title" : clean(title),
                        "content" : "",
                        "subsections" : [],
                    })
                elif rank == 3:
                    if not len(sections):
                        sections.append({
                            "title" : "",
                            "content" : "",
                            "subsections" : [],
                        })
                    sections[-1]["subsections"].append({
                        "title" : clean(title),
                        "content" : "",
                    })
            else:
                if rank == 2:
                    sections[-1]["content"] += line + "\n"
                elif rank == 3:
                    sections[-1]["subsections"][-1]["content"] += line + "\n"

        return sections

    def get_context(self):
        context = ""
        for line in self.source.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith("="): break
            context += line + "\n"
        return context

    def get_title(self):
        return clean(self.title)

    def extract(self):
        docs = []

        title = self.get_title()
        context = self.get_context()
        sections = self.get_sections()

        docs.append(clean(context))

        for section in sections:
            if section["title"] in self.IGNORES_SECTIONS: continue
            docs.append(title + " " + clean(section["content"]))
            for subsection in section["subsections"]:
                docs.append(title + " " + clean(section["title"]) + " " + clean(subsection["content"]))
        return docs

docs_data = open("wikipedia_20220620_all_wiki.jsonl", "r", encoding="utf-8")
titles_data = open("wikipedia_20220620_all_titles.txt", "r", encoding="utf-8").readlines()

output_dir = "docs_test"
os.makedirs(output_dir, exist_ok=True)

ids = [*range(len(titles_data))]
chunk_parts = 20
chunk_size = len(ids) // chunk_parts
chunks = [ids[i * chunk_size:(i + 1) * chunk_size] for i in range((len(ids) + chunk_size - 1) // chunk_size )]
# chunks[-1].extend(ids[chunk_size * chunk_parts:])

links_dict = {}

print("Chunks: ", len(chunks))
for batch in range(len(chunks)):
    with open(f"{output_dir}/wikipedia_20220620_{batch}.jsonl", "w", encoding="utf-8") as f:
        for i in tqdm(range(len(chunks[batch])), desc="Processing"):
            # Read the line by line
            line = docs_data.readline()
            # Convert the line to a dictionary
            data = json.loads(line)

            title = data["title"]
            text = data["text"].splitlines()

            texts = WikiDoc(title, text).extract()
            i_links = {}

            for i_line, line in enumerate(texts):

                links, line = get_and_replace_internal_links(line)

                doc = {
                    "id": title + "_" + str(i_line),
                    "contents": line
                }

                links = {link: {'start': start_id, 'end': end_id} for link, start_id, end_id in links}

                i_links[i_line] = links

                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            links_dict[title] = i_links

with open("wikipedia_20220620_all_links.json", "w", encoding="utf-8") as f:
    json.dump(links_dict, f, ensure_ascii=False)
