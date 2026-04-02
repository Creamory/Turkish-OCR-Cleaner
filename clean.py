"""
clean.py — Turkish news CSV cleaning pipeline
Input:  example.csv  (cp1254-encoded)
Output: data/clean.csv (utf-8-sig)
Charts: data/eda/*.png
"""

import os
import re
import hashlib

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE     = "subat14-mart14.csv"
OUTPUT_FILE    = "data/cleanheavysiyasi.csv"
EDA_DIR        = "data/eda3"
INPUT_ENCODING = "utf-8-sig"

os.makedirs("data", exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# ── Acronym whitelist ─────────────────────────────────────────────────────────
ACRONYMS = {
    "ABD", "AFAD", "AKP", "BB", "BDDK", "BM", "BOTAŞ", "CHP", "DHA",
    "DOKAP", "EKAP", "EYF", "GSYİH", "HDP", "İHA", "İYİP", "KKTC",
    "KDV", "LGS", "MHP", "MİT", "MKE", "NATO", "OSB", "ÖSYM", "PTT",
    "RTÜK", "SGK", "SPK", "TBMM", "TCK", "TCDD", "TFF", "THY", "TİM",
    "TOKİ", "TOBB", "TRT", "TÜİK", "TÜSİAD", "YKS", "YÖK", "YSK",
    "TL", "AVM", "ÇED", "KOBİ"
}

# Load additional acronyms from custom_acronyms.txt if it exists
_ACRONYMS_FILE = "data/custom_acronyms.txt"
if os.path.exists(_ACRONYMS_FILE):
    with open(_ACRONYMS_FILE, encoding="utf-8") as _f:
        _extra = {
            line.strip().upper()
            for line in _f
            if line.strip() and not line.startswith("#")
        }
    ACRONYMS |= _extra
    print(f"[acronyms]       loaded {len(_extra)} extra acronym(s) from {_ACRONYMS_FILE}")

# Turkish conjunctions/prepositions that stay lowercase mid-title
LOWERCASE_WORDS = {
    "ve", "ile", "de", "da", "ki", "için", "veya", "hem", "ya", "bir",
    "mi", "mı", "mu", "mü"
}

# ── Helper: print step summary ────────────────────────────────────────────────
def report(step: str, df: pd.DataFrame, prev_len: int) -> None:
    dropped = prev_len - len(df)
    tag = f"(-{dropped})" if dropped else "(no change)"
    print(f"[{step:<16}] {len(df):>6} rows  {tag}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, encoding=INPUT_ENCODING)
print(f"[{'load':<16}] {len(df):>6} rows  (raw input)")
prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Drop rows with null or empty title / content
# ═══════════════════════════════════════════════════════════════════════════════
df = df[df["title"].notna() & df["content"].notna()]
df = df[df["title"].str.strip().ne("") & df["content"].str.strip().ne("")]
report("null/empty", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Drop rows where content is under 200 characters
# ═══════════════════════════════════════════════════════════════════════════════
df = df[df["content"].str.len() >= 200]
report("short content", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Drop rows with OCR artifacts
# ═══════════════════════════════════════════════════════════════════════════════
_OCR_RE = re.compile(
    r"\w\^"
    r"|\^\w"
    r"|\^{2,}"
    r"|\*\^"
    r"|[a-zğüşıöçâîû][A-ZĞÜŞİÖÇÂÎÛ]"
)

def has_ocr_artifacts(text: str) -> bool:
    return bool(_OCR_RE.search(str(text)))

ocr_mask = df["title"].apply(has_ocr_artifacts) | df["content"].apply(has_ocr_artifacts)
print(f"  * OCR artifacts found in {ocr_mask.sum()} rows (title or content)")
df = df[~ocr_mask]
report("OCR artifacts", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3.5 — Normalize whitespace in content
# Collapse tabs and multiple spaces to single space
# ═══════════════════════════════════════════════════════════════════════════════
df["content"] = df["content"].apply(
    lambda x: re.sub(r"[\t ]{2,}", " ", str(x)).strip()
)
report("whitespace", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Strip agency boilerplate from titles
# ═══════════════════════════════════════════════════════════════════════════════
_AGENCY_TAG = re.compile(
    r"\(\s*(?:AA|DHA|[İI]HA|AP|Reuters?|ÖZEL|ANKA|TRT|UHA|AFP|CHA)\s*\)",
    re.IGNORECASE,
)
_AGENCY_PREFIX = re.compile(
    r"^(?:HABER\s+MERKEZ[İI]\s*[-–]|ÖZEL\s*[-–]|\(ÖZEL\)\s*[-–]?)\s*",
    re.IGNORECASE,
)

def strip_agency(title: str) -> str:
    title = _AGENCY_TAG.sub("", str(title))
    title = _AGENCY_PREFIX.sub("", title)
    return title.strip(" \t-–()")

df["title"] = df["title"].apply(strip_agency)
report("agency strip", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Turkish-aware title-casing for ALL-CAPS titles
# ═══════════════════════════════════════════════════════════════════════════════
_TR_TO_LOWER = str.maketrans("Iİ", "ıi")
_TR_TO_UPPER = str.maketrans("iı", "İI")

def turkish_lower(text: str) -> str:
    return text.translate(_TR_TO_LOWER).lower()

def turkish_title_case(title: str) -> str:
    if not title.isupper():
        return title
    result = []
    words = turkish_lower(title).split()
    for i, word in enumerate(words):
        if not word:
            continue

        # Handle apostrophe-suffixed tokens like ABD'den or NATO'nun
        apos_match = re.search(r"['\u2019]", word)
        if apos_match:
            apos_idx = apos_match.start()
            stem     = word[:apos_idx]
            suffix   = word[apos_idx:]
            if stem.upper() in ACRONYMS:
                result.append(stem.upper() + turkish_lower(suffix))
                continue

        # Check bare word against acronym set
        bare = word.strip("\"'""''()[].,;:!?-–—")
        if bare.upper() in ACRONYMS:
            result.append(bare.upper())
            continue

        # Mid-title conjunction — keep lowercase
        if i > 0 and bare in LOWERCASE_WORDS:
            result.append(word)
            continue

        # Default: capitalize first letter with Turkish rules
        first = word[0].translate(_TR_TO_UPPER).upper()
        result.append(first + word[1:])

    return " ".join(result)

df["title"] = df["title"].apply(turkish_title_case)
report("title case", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Deduplicate by uuid, then by content MD5 hash
# ═══════════════════════════════════════════════════════════════════════════════
before = len(df)
df = df.drop_duplicates(subset="uuid", keep="first")
uuid_dropped = before - len(df)
print(f"  * {uuid_dropped} uuid duplicates dropped")

df["_hash"] = df["content"].apply(
    lambda x: hashlib.md5(x.encode("utf-8", errors="replace")).hexdigest()
)
before = len(df)
df = df.drop_duplicates(subset="_hash", keep="first")
hash_dropped = before - len(df)
df = df.drop(columns=["_hash"])
print(f"  * {hash_dropped} content-hash duplicates dropped")

report("dedup", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Drop rows where the title exceeds 15 words
# ═══════════════════════════════════════════════════════════════════════════════
word_counts     = df["title"].str.split().str.len()
long_title_mask = word_counts > 15
print(f"  * {long_title_mask.sum()} rows have titles longer than 15 words")
df = df[~long_title_mask]
report("title length", df, prev); prev = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nSaved {len(df)} clean rows -> {OUTPUT_FILE}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving EDA charts...")

def save_hist(series: pd.Series, xlabel: str, title: str,
              filename: str, color: str = "steelblue") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series.dropna(), bins=40, color=color, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(EDA_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  * {path}")

save_hist(
    df["title"].str.split().str.len(),
    xlabel="Word count",
    title="Title length distribution (words)",
    filename="title_length.png",
    color="steelblue",
)

save_hist(
    df["content"].str.len(),
    xlabel="Character count",
    title="Content length distribution (characters)",
    filename="content_length.png",
    color="darkorange",
)

if "category" in df.columns and df["category"].notna().any():
    cat_counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(max(6, len(cat_counts) * 0.9), 4))
    cat_counts.plot(kind="bar", ax=ax, color="seagreen", edgecolor="white")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_title("Article count by category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(EDA_DIR, "category_breakdown.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  * {path}")

print("\nDone.")
