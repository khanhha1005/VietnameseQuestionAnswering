import re 
import os
import json 
from tqdm import tqdm

acceptedNamespaces = ['w', 'wiktionary', 'wikt']
# match tail after wikilink
tailRE = re.compile('\w+')
syntaxhighlight = re.compile('&lt;syntaxhighlight .*?&gt;(.*?)&lt;/syntaxhighlight&gt;', re.DOTALL)

keepLinks = True

def findBalanced(text, openDelim, closeDelim):
    """
    Assuming that text contains a properly balanced expression using
    :param openDelim: as opening delimiters and
    :param closeDelim: as closing delimiters.
    :return: an iterator producing pairs (start, end) of start and end
    positions in text containing a balanced expression.
    """
    openPat = '|'.join([re.escape(x) for x in openDelim])
    # patter for delimiters expected after each opening delimiter
    afterPat = {o: re.compile(openPat + '|' + c, re.DOTALL) for o, c in zip(openDelim, closeDelim)}
    stack = []
    start = 0
    cur = 0
    # end = len(text)
    startSet = False
    startPat = re.compile(openPat)
    nextPat = startPat
    while True:
        next = nextPat.search(text, cur)
        if not next:
            return
        if not startSet:
            start = next.start()
            startSet = True
        delim = next.group(0)
        if delim in openDelim:
            stack.append(delim)
            nextPat = afterPat[delim]
        else:
            opening = stack.pop()
            # assert opening == openDelim[closeDelim.index(next.group(0))]
            if stack:
                nextPat = afterPat[stack[-1]]
            else:
                yield start, next.end()
                nextPat = startPat
                start = next.end()
                startSet = False
        cur = next.end()


def makeInternalLink(title, label):
    colon = title.find(':')
    if colon > 0 and title[:colon] not in acceptedNamespaces:
        return ''
    if colon == 0:
        # drop also :File:
        colon2 = title.find(':', colon + 1)
        if colon2 > 1 and title[colon + 1:colon2] not in acceptedNamespaces:
            return ''
    if keepLinks:
        if len(title) > 1:
          title = title[0].upper() + title[1:]
        elif len(title) == 1:
          title = title[0].upper()
        title = "_".join(title.split())
        return '[[wiki/%s|%s]]' % (title, label)
    else:
        return label

def replaceInternalLinks(text):
    """
    Replaces external links of the form:
    [[title |...|label]]trail

    with title concatenated with trail, when present, e.g. 's' for plural.
    """
    # call this after removal of external links, so we need not worry about
    # triple closing ]]].
    cur = 0
    res = ''
    for s, e in findBalanced(text, ['[['], [']]']):
        m = tailRE.match(text, e)
        if m:
            trail = m.group(0)
            end = m.end()
        else:
            trail = ''
            end = e
        inner = text[s + 2:e - 2]
        # find first |
        pipe = inner.find('|')
        if pipe < 0:
            title = inner
            label = title
        else:
            title = inner[:pipe].rstrip()
            # find last |
            curp = pipe + 1
            for s1, e1 in findBalanced(inner, ['[['], [']]']):
                last = inner.rfind('|', curp, s1)
                if last >= 0:
                    pipe = last  # advance
                curp = e1
            label = inner[pipe + 1:].strip()
        res += text[cur:s] + makeInternalLink(title, label) + trail
        cur = end
    return res + text[cur:]

def main():
    data_dir = "dump"
    sub_dirs = os.listdir(data_dir)
    wiki_files = [ os.path.join(data_dir, sub_dir, file) 
                    for sub_dir in sub_dirs 
                    for file in os.listdir(os.path.join(data_dir, sub_dir)) if file.startswith("wiki_") ]
    print("Found %d wiki files" % len(wiki_files))
    
    titles = open("wikipedia_20220620_cleaned/wikipedia_20220620_all_titles.txt", "r", encoding="utf-8").readlines()
    titles = [title.strip() for title in titles]
    print("Found %d titles" % len(titles))

    wiki_dumps = open("wikipedia_20220620_all_wiki.jsonl", "w", encoding="utf-8")

    bar = tqdm(total=len(wiki_files))

    for wiki_file in wiki_files:
        bar.set_description("Processing %s" % wiki_file)

        with open(wiki_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                id = line["id"]
                revid = line["revid"]
                url = line["url"]
                title = line["title"]
                text = line["text"]

                # if title not in titles:
                #     continue

                # text = replaceInternalLinks(text)
                # text = syntaxhighlight.sub(r'\1', text)

                data = {
                    "id": id,
                    "revid": revid,
                    "url": url,
                    "title": title,
                    "text": text
                }

                wiki_dumps.write(json.dumps(data, ensure_ascii=False) + "\n")
                wiki_dumps.flush()

        bar.update(1)

if __name__ == '__main__':
    main()
