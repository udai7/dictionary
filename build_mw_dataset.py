"""
Build a large Sanskrit -> English dataset from the Monier-Williams dictionary
(Cologne Digital Sanskrit Dictionaries, ~186k entries).

Step 1: download the raw MW source if not already present.
Step 2: parse each entry -> (sanskrit headword in IAST, english gloss, pos).
Step 3: write data/dictionary.csv  (columns: sanskrit, english, pos)

train.py automatically uses data/dictionary.csv when it exists.

Run:
    python build_mw_dataset.py
"""
import csv
import os
import re
import urllib.request

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "mw_raw.txt")
OUT = os.path.join(HERE, "data", "dictionary.csv")
URL = ("https://raw.githubusercontent.com/sanskrit-lexicon/csl-orig/"
       "master/v02/mw/mw.txt")

# --- SLP1 -> IAST transliteration -----------------------------------------
# The MW <k1> headwords are encoded in SLP1 (one ASCII char per phoneme).
SLP1_IAST = {
    "A": "ā", "I": "ī", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ",
    "E": "ai", "O": "au", "M": "ṃ", "H": "ḥ", "~": "m̐",
    "K": "kh", "G": "gh", "N": "ṅ", "C": "ch", "J": "jh", "Y": "ñ",
    "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ",
    "T": "th", "D": "dh", "P": "ph", "B": "bh",
    "S": "ś", "z": "ṣ", "L": "ḻ", "|": "", "'": "’",
}


def slp1_to_iast(s: str) -> str:
    out = []
    for ch in s:
        if ch in SLP1_IAST:
            out.append(SLP1_IAST[ch])
        elif ch.isdigit():
            continue  # drop homonym numbers
        else:
            out.append(ch)  # a i u e o k g c j t d n p b m y r l v s h stay as-is
    return "".join(out)


POS_MAP = {
    "m.": "noun", "f.": "noun", "n.": "noun", "nf.": "noun", "fn.": "noun",
    "nm.": "noun", "mn.": "noun", "m.f.": "noun", "m": "noun",
    "mfn.": "adjective", "mf.": "adjective", "fn.adj.": "adjective",
    "ind.": "indeclinable",
}

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
K1_RE = re.compile(r"<k1>([^<]*)")
# Non-greedy so it spans nested tags, e.g. <lex>mf(<s>A</s>)n.</lex>.
LEX_RE = re.compile(r"<lex>(.*?)</lex>")
PAREN_RE = re.compile(r"\([^)]*\)")

# Leading POS abbreviation in the gloss, including the gendered adjective form
# like "mf(A)n." or "mf(A/)n." that precedes the actual definition.
POS_ABBR_RE = re.compile(
    r"^(mf\([^)]*\)n|mfn|mf|m|f|n|nf|fn|nm|ind)\.\s*", re.I)

# Verb roots: MW marks a conjugation class with "<ab>cl.</ab>". Each sense of
# the meaning is introduced by a self-closing <div .../> tag, e.g.
#   <div n="to"/>to go, move, ... <ls>RV.</ls>
# We take the text after each <div> up to the first citation (<ls>) tag.
CL_RE = re.compile(r"<ab>cl\.</ab>")
DIV_GLOSS_RE = re.compile(r"<div[^>]*/>\s*(.*?)(?=<ls|<div|$)", re.S)


def verb_gloss(ent: str) -> str:
    """Collect the meaning lines of a verb-root entry into one clean gloss."""
    senses = []
    for m in DIV_GLOSS_RE.finditer(ent):
        seg = TAG_RE.sub("", m.group(1))
        seg = WS_RE.sub(" ", seg).strip(" ,.;:")
        if seg and seg.lower().startswith("to "):
            senses.append(seg)
    return ", ".join(dict.fromkeys(senses))   # dedupe, preserve order


def normalize_lex(raw: str) -> str:
    """Turn a raw <lex> body into a canonical POS abbreviation.

    Strips inner markup tags, the parenthetical feminine-stem hint, and the
    number/case markers so that `mf(A)n.`, `m.pl.`, `m.du.` etc. fold onto
    their base form (`mfn.`, `m.`).
    """
    s = TAG_RE.sub("", raw)        # mf(<s>A</s>)n. -> mf(A)n.
    s = PAREN_RE.sub("", s)        # mf(A)n.        -> mfn.
    s = WS_RE.sub("", s)           # drop stray spaces, e.g. "mn (?)."
    for suff in ("pl.", "du.", "sg.", "pl", "du", "sg"):
        s = s.replace(suff, "")    # m.pl. -> m. ; mfn.pl. -> mfn.
    return s.strip()


def clean_english(body: str) -> str:
    # Drop the <L>...header line(s); keep only definition lines.
    lines = [ln for ln in body.splitlines() if "<L>" not in ln]
    text = " ".join(lines)
    # The "¦" separates the SLP1 headword from the actual gloss -> keep gloss.
    if "¦" in text:
        text = text.split("¦", 1)[1]
    text = TAG_RE.sub("", text)          # strip all remaining markup
    text = text.replace("&c.", "etc.")   # MW abbreviation for "et cetera"
    text = text.replace("&", "")
    text = WS_RE.sub(" ", text).strip(" ,.;:")
    text = POS_ABBR_RE.sub("", text)     # drop a leading "mf(A)n." / "m." etc.
    return text.strip(" ,.;:")


def download():
    if os.path.exists(RAW):
        return
    print("Downloading Monier-Williams source (~49 MB)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "mw-dl"})
    data = urllib.request.urlopen(req, timeout=180).read()
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    with open(RAW, "wb") as f:
        f.write(data)
    print("  saved", RAW)


def parse():
    text = open(RAW, encoding="utf-8", errors="replace").read()
    # Split into entries on the <L>...<LEND> markers.
    entries = re.split(r"<LEND>", text)
    rows = []
    seen = set()
    for ent in entries:
        m = K1_RE.search(ent)
        if not m:
            continue
        head = slp1_to_iast(m.group(1)).strip().lower()
        if not head or len(head) > 40:
            continue
        lex = LEX_RE.search(ent)
        pos = POS_MAP.get(normalize_lex(lex.group(1))) if lex else None

        if pos:
            eng = clean_english(ent)
        elif CL_RE.search(ent):
            # No nominal POS tag, but a verb-class marker -> it's a verb root.
            pos = "verb"
            eng = verb_gloss(ent)
        else:
            continue

        if not eng or len(eng) < 2:
            continue
        eng = eng[:200]  # keep glosses reasonable
        key = (head, pos)
        if key in seen:
            continue
        seen.add(key)
        rows.append((head, eng, pos))
    return rows


def main():
    download()
    rows = parse()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sanskrit", "english", "pos"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} entries to {OUT}")
    from collections import Counter
    print(Counter(r[2] for r in rows))


if __name__ == "__main__":
    main()
