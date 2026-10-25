import re
from collections import Counter
from pathlib import Path
import time
from urllib.parse import unquote

import requests
import torch


# Wikimedia and Gutenberg both ask bots to identify themselves honestly
_USER_AGENT = "tiny-gpt/0.1 (educational project; https://github.com/yekta-kaya/tiny-gpt-pytorch)"

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_TAIL_SECTIONS = re.compile(
    r"\n==\s*(See also|References|Notes|Citations|Sources|Further reading|External links)\s*==",
    re.IGNORECASE,
)


def fetch_wikipedia_text(urls, verbose=True):
    # MediaWiki API instead of scraping article HTML: it's what Wikimedia's
    # robot policy asks for, and it doesn't get rate-limited with 403s.
    headers = {"User-Agent": _USER_AGENT}
    all_texts = []

    for url in urls:
        title = unquote(url.rsplit("/wiki/", 1)[-1])
        try:
            response = requests.get(_WIKI_API, headers=headers, timeout=20, params={
                "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                "titles": title, "format": "json", "formatversion": 2,
            })
            response.raise_for_status()
            page_text = response.json()["query"]["pages"][0].get("extract")
            if not page_text:
                raise ValueError("page not found")

            # drop reference/link sections at the end, then the "== Heading ==" lines
            tail = _WIKI_TAIL_SECTIONS.search(page_text)
            if tail:
                page_text = page_text[:tail.start()]
            page_text = re.sub(r"^=+.*=+\s*$", "", page_text, flags=re.MULTILINE)
            page_text = " ".join(page_text.split())
            all_texts.append(page_text)

            if verbose:
                print(f"Fetched: {title} ({len(page_text):,} chars)")

        except Exception as e:
            print(f"  Failed {url}: {e}")
        time.sleep(1)
    return " ".join(all_texts)


_GUTENBERG_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def fetch_gutenberg_text(book_ids, verbose=True):
    headers = {"User-Agent": _USER_AGENT}
    all_texts = []

    for book_id in book_ids:
        fetched = False
        for template in _GUTENBERG_URL_TEMPLATES:
            url = template.format(id=book_id)
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code != 200:
                    continue
                text = response.text

                start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
                end   = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",   text, re.IGNORECASE)
                if start:
                    text = text[start.end():]
                if end:
                    text = text[:end.start()]

                text = " ".join(text.split())
                all_texts.append(text)

                if verbose:
                    print(f"Fetched Gutenberg #{book_id}: {len(text):,} chars")
                fetched = True
                break

            except Exception:
                continue

        if not fetched:
            print(f"  Failed Gutenberg #{book_id}: could not retrieve from any URL")
        time.sleep(2)

    return " ".join(all_texts)


_DEFAULT_WIKIPEDIA_URLS = [
    # Dogs & breeds
    "https://en.wikipedia.org/wiki/Dog",
    "https://en.wikipedia.org/wiki/Dog_behavior",
    "https://en.wikipedia.org/wiki/Dog_training",
    "https://en.wikipedia.org/wiki/Labrador_Retriever",
    "https://en.wikipedia.org/wiki/Golden_Retriever",
    "https://en.wikipedia.org/wiki/German_Shepherd",
    "https://en.wikipedia.org/wiki/Poodle",
    "https://en.wikipedia.org/wiki/Bulldog",
    "https://en.wikipedia.org/wiki/Beagle",
    "https://en.wikipedia.org/wiki/Border_Collie",
    "https://en.wikipedia.org/wiki/Siberian_Husky",
    # Canid family & wild relatives
    "https://en.wikipedia.org/wiki/Wolf",
    "https://en.wikipedia.org/wiki/Dingo",
    "https://en.wikipedia.org/wiki/Canidae",
    "https://en.wikipedia.org/wiki/Coyote",
    "https://en.wikipedia.org/wiki/Fox",
    # Animal cognition & domestication
    "https://en.wikipedia.org/wiki/Domestication_of_the_dog",
    "https://en.wikipedia.org/wiki/Animal_cognition",
    "https://en.wikipedia.org/wiki/Animal_communication",
    "https://en.wikipedia.org/wiki/Pack_(canine)",
    # Broader animal world
    "https://en.wikipedia.org/wiki/Cat",
    "https://en.wikipedia.org/wiki/Horse",
    "https://en.wikipedia.org/wiki/Wildlife",
]

# 215  - The Call of the Wild   (Jack London, 1903)
# 910  - White Fang             (Jack London, 1906)
# 2852 - The Hound of the Baskervilles (Conan Doyle, 1902)
# 8177 - Beautiful Joe          (Marshall Saunders, 1893)
# 271  - Black Beauty           (Anna Sewell, 1877)
# 236  - The Jungle Book        (Rudyard Kipling, 1894)
# 289  - The Wind in the Willows (Kenneth Grahame, 1908)
_DEFAULT_GUTENBERG_IDS = [215, 910, 2852, 8177, 271, 236, 289]


def gather_data(
    wikipedia_urls=None,
    gutenberg_ids=None,
    verbose=True,
):
    parts = []

    urls = wikipedia_urls if wikipedia_urls is not None else _DEFAULT_WIKIPEDIA_URLS
    if verbose:
        print(f"=== Wikipedia ({len(urls)} pages) ===")
    wiki_text = fetch_wikipedia_text(urls, verbose=verbose)
    if wiki_text:
        parts.append(wiki_text)

    ids = gutenberg_ids if gutenberg_ids is not None else _DEFAULT_GUTENBERG_IDS
    if verbose:
        print(f"\n=== Project Gutenberg ({len(ids)} books) ===")
    gut_text = fetch_gutenberg_text(ids, verbose=verbose)
    if gut_text:
        parts.append(gut_text)

    combined = " <doc> ".join(parts)
    if verbose:
        print(f"\nTotal gathered: {len(combined):,} chars")
    return combined



def format_data(text, seq_len=64, min_freq=3, stride=None):
    # stride = gap between window starts. Defaults to seq_len (no overlap):
    # stride=1 puts every token in seq_len windows, so one "epoch" is really
    # ~seq_len passes over the corpus and the model overfits within an epoch.
    stride = stride or seq_len

    # ── 1. lowercase + normalize whitespace ──────────────────────
    text = text.lower()
    text = text.replace("\n", " ")

    # ── 2. strip/separate punctuation properly ───────────────────
    # separate . , ! ? ; : from words so "dog." → "dog ."
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    # remove everything else that isn't a letter, digit, or kept punctuation
    text = re.sub(r"[^a-z0-9.,!?;: ]", " ", text)
    # collapse multiple spaces
    text = " ".join(text.split())

    tokens = text.split()

    # ── 3. frequency cutoff — drop words seen fewer than min_freq times ──
    counts = Counter(tokens)
    tokens = [t if counts[t] >= min_freq else "<unk>" for t in tokens]

    # ── 4. build vocab ───────────────────────────────────────────
    vocab = sorted(set(tokens))
    stoi  = {w: i for i, w in enumerate(vocab)}
    itos  = {i: w for w, i in stoi.items()}

    encoded = [stoi[t] for t in tokens]

    # ── 5. sequences ─────────────────────────────────────────────
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_len, stride):
        inputs.append(encoded[i     : i + seq_len])
        targets.append(encoded[i + 1: i + seq_len + 1])

    input_ids  = torch.tensor(inputs,  dtype=torch.long)
    target_ids = torch.tensor(targets, dtype=torch.long)

    print(f"Vocab size : {len(vocab):,}")
    print(f"Sequences  : {len(inputs):,}  (seq_len={seq_len}, stride={stride})")
    print(f"<unk> rate : {tokens.count('<unk>') / len(tokens) * 100:.2f}%")

    return input_ids, target_ids, vocab, stoi, itos


def train_val_split(input_ids, target_ids, val_frac=0.1, seed=42):
    # Random split over windows so val covers every source, not just whichever
    # book happens to come last. Only valid with non-overlapping windows
    # (stride >= seq_len); with overlap, val windows would repeat train text.
    gen  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(input_ids.size(0), generator=gen)
    n_val = int(val_frac * input_ids.size(0))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return input_ids[train_idx], target_ids[train_idx], input_ids[val_idx], target_ids[val_idx]


if __name__ == "__main__":
    text = gather_data()

    out_path = Path(__file__).resolve().parent.parent / "data" / "dog_corpus.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nSaved {len(text):,} chars to {out_path}")