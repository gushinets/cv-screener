"""Independent resume BM25 and dense retrieval, fused at document level."""
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r'c\+\+|c#|\.net|[^\W_]+(?:[.+-][^\W_]+)*', text.casefold().replace('ё', 'е'))


def rank_documents(query, dense, chunks, limit=25):
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.metadata['parent']].append(chunk)
    if not grouped:
        return []

    representatives, aliases, by_text, texts = {}, {}, {}, []
    for name in sorted(grouped, key=str.casefold):
        parts = sorted(grouped[name], key=lambda d: d.metadata.get('chunk', 0))
        text = '\n\n'.join(part.page_content for part in parts)
        canonical = by_text.setdefault(text, name)
        aliases[name] = canonical
        representatives[name] = parts[0]
        if canonical == name:
            texts.append((name, tokenize(text)))

    ordered_dense = sorted(dense, key=lambda d: (-round(d.score, 6),
                           d.metadata['parent'].casefold(), d.metadata.get('chunk', 0)))
    dense_names = list(dict.fromkeys(aliases[d.metadata['parent']] for d in ordered_dense
                                    if d.metadata['parent'] in aliases))
    lexical_names = []
    if any(tokens for _, tokens in texts):
        scores = BM25Okapi([tokens for _, tokens in texts]).get_scores(tokenize(query))
        lexical_names = [texts[i][0] for i in sorted(range(len(texts)),
                         key=lambda i: (-scores[i], texts[i][0].casefold()))[:100] if scores[i] > 0]

    scores, dense_rank, lexical_rank = {}, {}, {}
    for names, ranks in [(dense_names, dense_rank), (lexical_names, lexical_rank)]:
        for rank, name in enumerate(names, 1):
            ranks[name] = rank
            scores[name] = scores.get(name, 0) + .5 / (2 + rank)
    ranked = sorted(scores, key=lambda name: (-scores[name], dense_rank.get(name, float('inf')),
                    lexical_rank.get(name, float('inf')), name.casefold()))

    pinned, preferred = [], {}
    if re.search(r'\.(?:pdf|docx|txt)\b', query, re.IGNORECASE):
        for name in sorted(grouped, key=str.casefold):
            if name.casefold() in query.casefold():
                canonical = aliases[name]
                pinned.append(canonical)
                preferred.setdefault(canonical, representatives[name])
    ranked = list(dict.fromkeys(pinned + ranked))[:limit]
    return [preferred.get(name, representatives[name]) for name in ranked]
