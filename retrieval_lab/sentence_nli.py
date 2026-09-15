"""Sentence-level entailment interface with a lexical fallback baseline."""
from __future__ import annotations
import re

class SentenceNLI:
    def __init__(self, model_path=None):
        self.model_path=model_path; self.pipe=None
        if model_path:
            from transformers import pipeline
            self.pipe=pipeline('text-classification',model=str(model_path),tokenizer=str(model_path),local_files_only=True)
    def predict(self, claim, sentences):
        if self.pipe:
            return self.pipe([{'text':s,'text_pair':claim} for s in sentences],truncation=True)
        q=set(re.findall(r'[a-z0-9]+',claim.lower())); out=[]
        neg={'no','not','never','lack','without','fails','failed'}
        for s in sentences:
            t=set(re.findall(r'[a-z0-9]+',s.lower())); overlap=len(q&t)/max(len(q),1); conflict=bool((q&neg)^(t&neg))
            out.append({'label':'CONTRADICTION' if conflict and overlap>.3 else 'ENTAILMENT' if overlap>.45 else 'NEUTRAL','score':overlap})
        return out
