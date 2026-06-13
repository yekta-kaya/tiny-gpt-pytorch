import re
import requests
import torch
from bs4 import BeautifulSoup


def fetch_wikipedia_text(urls, verbose=True):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LLMExperiments/1.0)"}
    all_texts = []

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()

            if verbose:
                print(f"Fetched: {url}")

            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", {"id": "mw-content-text"})
            paragraphs = (content or soup).find_all("p")
            paragraphs = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]

            if verbose:
                print(f"  Paragraphs: {len(paragraphs)}")

            page_text = " ".join(paragraphs)
            page_text = " ".join(page_text.split())
            all_texts.append(page_text)

        except Exception as e:
            print(f"  Failed {url}: {e}")

    return " ".join(all_texts)


_GUTENBERG_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def fetch_gutenberg_text(book_ids, verbose=True):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LLMExperiments/1.0)"}
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


def format_data(text, seq_len=64):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace(".", " .")
    text = text.replace(",", " ,")
    text = " ".join(text.split())

    tokens = text.split()
    vocab = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}

    encoded = [stoi[w] for w in tokens]

    inputs, targets = [], []
    for i in range(len(encoded) - seq_len):
        inputs.append(encoded[i : i + seq_len])
        targets.append(encoded[i + 1 : i + seq_len + 1])

    input_ids  = torch.tensor(inputs,  dtype=torch.long)
    target_ids = torch.tensor(targets, dtype=torch.long)

    print(f"Vocab size : {len(vocab):,}")
    print(f"Sequences  : {len(inputs):,}  (seq_len={seq_len})")
    return input_ids, target_ids, vocab, stoi, itos
