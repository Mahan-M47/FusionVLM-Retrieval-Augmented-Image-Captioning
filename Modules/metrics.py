import nltk
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from pycocoevalcap.cider.cider import Cider


# -----------------------
# Tokenization helper
# -----------------------
def tokenize(sentences):
    return [s.lower().split() for s in sentences]


# -----------------------
# BLEU (1–4)
# -----------------------
def compute_bleu(preds, refs):
    preds_tok = tokenize(preds)
    refs_tok = [[r.lower().split() for r in ref_list] for ref_list in refs]

    smooth = SmoothingFunction().method4

    return {
        "BLEU-1": corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-2": corpus_bleu(refs_tok, preds_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "BLEU-3": corpus_bleu(refs_tok, preds_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }


# -----------------------
# METEOR
# -----------------------
# def compute_meteor(preds, refs):
#     scores = []
#     for p, r in zip(preds, refs):
#         scores.append(meteor_score(r, p))
#     return np.mean(scores)

def compute_meteor(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        hyp = tokenize([p])[0]        # List[str]
        ref = tokenize(r)             # List[List[str]]
        scores.append(meteor_score(ref, hyp))
    return np.mean(scores)

# -----------------------
# ROUGE-L
# -----------------------
def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for p, r in zip(preds, refs):
        s = max(scorer.score(ref, p)["rougeL"].fmeasure for ref in r)
        scores.append(s)

    return np.mean(scores)


# -----------------------
# CIDEr (COCO-style)
# -----------------------
def compute_cider(preds, refs):
    """
    CIDEr expects dicts:
    {id: [caption]}
    """
    gts = {i: refs[i] for i in range(len(refs))}
    res = {i: [preds[i]] for i in range(len(preds))}

    cider = Cider()
    score, _ = cider.compute_score(gts, res)
    return score


# -----------------------
# Master evaluation
# -----------------------
def setup_nltk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # recommended for newer WordNet versions
    
def evaluate_captioning(preds, refs):
    assert len(preds) == len(refs)

    results = {}
    results.update(compute_bleu(preds, refs))
    results["METEOR"] = compute_meteor(preds, refs)
    results["ROUGE-L"] = compute_rouge_l(preds, refs)
    results["CIDEr"] = compute_cider(preds, refs)

    return results
