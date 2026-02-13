### Hi there 👋

I am Ifeanyi Anthony Omeife, a data scientist and machine learning engineer based in the United Kingdom.  M.sc Applied Statistics @ University of St Andrews

📫 How to reach me: iomeife@gmail.com <br>
<br>
⚡ Fun fact: I love rapping and I occassionally take singing breaks when studying lol. <br>

🤔 I am interested in  
- Computer Vision
- Machine Learning
- Language Models
- Reinforcement Learning.<br>

😄 One of my favorite quotes is  'dare to win!.



<!--
**Ifeanyi-omeck/Ifeanyi-omeck** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- .
- 
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 
- 😄 Pronouns: ...
- 

-->


import re
import pandas as pd
from rapidfuzz import fuzz

def _clean_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z\s'-]", "", s)          # keep letters/spaces/'/-
    s = re.sub(r"\s+", " ", s)
    return s

def _clean_email(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    return s

def _email_local_domain(email: str):
    if not email or "@" not in email:
        return "", ""
    local, domain = email.split("@", 1)
    return local, domain

def fuzzy_match_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    a_first="first_name", a_last="last_name", a_email="email",
    b_first="first_name", b_last="last_name", b_email="email",
    min_score=80,
    top_n_candidates=200,
):
    """
    Returns a new dataframe with best match in df_b for each row in df_a.
    - Uses a simple blocking step: same email domain (when available), else same last-name initial.
    - Scores: first name, last name, email local part, and email domain (heavily weighted).
    """

    a = df_a.copy()
    b = df_b.copy()

    # Clean fields
    a["_fn"] = a[a_first].map(_clean_name)
    a["_ln"] = a[a_last].map(_clean_name)
    a["_em"] = a[a_email].map(_clean_email)

    b["_fn"] = b[b_first].map(_clean_name)
    b["_ln"] = b[b_last].map(_clean_name)
    b["_em"] = b[b_email].map(_clean_email)

    # Precompute blocks (domain and last initial)
    a["_em_local"], a["_em_domain"] = zip(*a["_em"].map(_email_local_domain))
    b["_em_local"], b["_em_domain"] = zip(*b["_em"].map(_email_local_domain))

    a["_ln_init"] = a["_ln"].str[:1].fillna("")
    b["_ln_init"] = b["_ln"].str[:1].fillna("")

    # Index B by email domain and last initial for fast candidate retrieval
    b_by_domain = {}
    for dom, grp in b.groupby("_em_domain", dropna=False):
        b_by_domain[dom] = grp

    b_by_lninit = {}
    for ch, grp in b.groupby("_ln_init", dropna=False):
        b_by_lninit[ch] = grp

    results = []

    def score_row(a_row, b_row) -> float:
        # Fuzzy scorers (0-100)
        fn = fuzz.token_sort_ratio(a_row["_fn"], b_row["_fn"])
        ln = fuzz.token_sort_ratio(a_row["_ln"], b_row["_ln"])

        # Email: treat domain as strong signal, local part as moderate
        dom = 100 if (a_row["_em_domain"] and a_row["_em_domain"] == b_row["_em_domain"]) else fuzz.ratio(a_row["_em_domain"], b_row["_em_domain"])
        loc = fuzz.ratio(a_row["_em_local"], b_row["_em_local"])

        # Weighted blend (tune as needed)
        # Domain matters a lot; last name matters a lot; first name moderate; email local moderate
        return 0.35 * ln + 0.20 * fn + 0.30 * dom + 0.15 * loc

    for a_idx, a_row in a.iterrows():
        # Blocking / candidate selection:
        # 1) If email domain exists, compare within same domain
        candidates = None
        if a_row["_em_domain"] in b_by_domain and a_row["_em_domain"] != "":
            candidates = b_by_domain[a_row["_em_domain"]]
        else:
            # 2) else fallback to last-name initial
            candidates = b_by_lninit.get(a_row["_ln_init"], b)

        # Limit candidate set for speed (optional, random/first N; better is smarter pre-filtering)
        if len(candidates) > top_n_candidates:
            candidates = candidates.head(top_n_candidates)

        best = None
        best_score = -1

        for b_idx, b_row in candidates.iterrows():
            s = score_row(a_row, b_row)
            if s > best_score:
                best_score = s
                best = (b_idx, b_row)

        if best is None:
            continue

        b_idx, b_row = best
        is_match = best_score >= min_score

        results.append({
            "a_index": a_idx,
            "b_index": b_idx,
            "match": is_match,
            "score": round(best_score, 2),

            # original fields (handy for inspection)
            "a_first": df_a.loc[a_idx, a_first],
            "a_last":  df_a.loc[a_idx, a_last],
            "a_email": df_a.loc[a_idx, a_email],
            "b_first": df_b.loc[b_idx, b_first],
            "b_last":  df_b.loc[b_idx, b_last],
            "b_email": df_b.loc[b_idx, b_email],
        })

    return pd.DataFrame(results).sort_values(["match", "score"], ascending=[False, False]).reset_index(drop=True)
