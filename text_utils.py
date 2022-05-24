import re
from typing import List, Tuple
import fasttext
from datasets.utils.download_manager import DownloadManager

import streamlit as st

prefix_length = len("__label__")

space_seperated_langs = ["en"]


@st.experimental_singleton
def get_fasttext_model():
  # pretrained_lang_model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
  pretrained_lang_model_url = "https://huggingface.co/julien-c/fasttext-language-id/resolve/main/lid.176.ftz"

  dl_manager = DownloadManager()
  return fasttext.load_model(dl_manager.download(pretrained_lang_model_url))


def detect_lang(sentence: str) -> str:
  model = get_fasttext_model()
  labels, probs = model.predict(sentence.replace("\n", " "), threshold=0.2)
  if len(labels) <= 0:
    return "unkown"
  return labels[0][prefix_length:]


def _split_punctuals(line: str) -> List[str]:
  matches = re.finditer(r"('|’|”|」|』)?(\.(?!\d)|\!|\?|。|！|？|…)+('|’|”|」|』|\s)?", line)
  matches = list(map(lambda x: x.end(), matches))
  filtered_matches = []
  for i, x in enumerate(matches):
    if (i - 1 >= 0 and x - matches[i - 1] <= 3) or (i + 1 < len(matches) and matches[i + 1] - x <= 3):
      continue
    if len(filtered_matches) <= 0 or x - filtered_matches[-1] > 10:
      filtered_matches.append(x)

  sentences = []
  i = 0
  for m in filtered_matches:
    sentences.append(line[i:m].strip())
    i = m
  if i < len(line):
    sentences.append(line[i:].strip())
  sentences = list(filter(lambda x: x, sentences))
  return sentences


def split_sentences(article: str) -> Tuple[List[str], List[str]]:
  lines = article.splitlines()
  processed, endls = [], []
  for line in lines:
    line = line.strip()
    if line:
      for sentence in _split_punctuals(line):
        processed.append(sentence)
        endls.append("")
    if len(endls):
      endls[-1] += "\n"
  return processed, endls


def join_sentences(sentences: List[str], endls: List[str], lang: str) -> str:
  sep = " " if lang in space_seperated_langs else ""
  result = ""
  for s, e in zip(sentences, endls):
    result += s
    result += e if e else sep
  return result


def prepare_simple(sentences: List[str], src_lang: str, trg_lang: str) -> List[str]:
  input_src = []
  for i in range(len(sentences)):
    src = f"{src_lang}2{trg_lang}: " + sentences[i]
    input_src.append(src)
  return input_src


def prepare_contexts(sentences: List[str], src_lang: str, trg_lang: str, max_length = 128, sep = "</s>")-> List[str]:
  input_src = []
  for i in range(len(sentences)):
    src = f"{src_lang}2{trg_lang}: " + sentences[i]
    src_before = sentences[max(0, i - 5):i]
    src_after = sentences[i + 1:i + 2]
    if len(src_before) > 0 and len(src) + len(src_before[-1]) < max_length:
        src = src_before.pop() + sep + src
    if len(src_after) > 0 and len(src) + len(src_after[0]) < max_length:
        src = src + sep + src_after[0]
    while len(src_before) > 0 and len(src) + len(src_before[-1]) < max_length:
        src = src_before.pop() + sep + src
    input_src.append(src)
  return input_src
