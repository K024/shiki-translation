import os
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration, Text2TextGenerationPipeline
import text_utils

import streamlit as st
from streamlit_option_menu import option_menu

langs = ["zh", "ja", "en"]

format_langs = dict(zh="Chinese", ja="Japanese", en="English", auto="Auto detect")


@st.experimental_singleton
def create_pipeline():
  model_config = dict(
    pretrained_model_name_or_path="K024/shiki-mt5-streaming",
    revision="main",
    use_auth_token=os.environ["ACCESS_TOKEN"] if "ACCESS_TOKEN" in os.environ else None,
  )

  pipeline = Text2TextGenerationPipeline(
    model=MT5ForConditionalGeneration.from_pretrained(**model_config),
    tokenizer=T5Tokenizer.from_pretrained(**model_config),
    device=0 if torch.cuda.is_available() else -1,
  )
  return pipeline


def translate(pipeline, src_lang, trg_lang, source, infer_config):
  if "CTX_TRANSLATION" in st.session_state:
    del st.session_state["CTX_TRANSLATION"]

  if not source:
    return

  if src_lang == "auto":
    src_lang = text_utils.detect_lang(source)

  if src_lang not in langs:
    st.error(f"Error: detected language '{src_lang}' is not supported.")
    return

  if src_lang == trg_lang:
    st.error("Error: source language is same as the target language.")
    return

  sentences, endls = text_utils.split_sentences(source)
  input_src = text_utils.prepare_contexts(sentences, src_lang, trg_lang)

  outputs = pipeline(input_src, **infer_config)
  results = [x["generated_text"] for x in outputs]
  joint_sentences = text_utils.join_sentences(results, endls, trg_lang)

  st.session_state["CTX_TRANSLATION"] = [src_lang, sentences, results, joint_sentences]


def translate_raw(pipeline, source, infer_config):
  input_src = [source]
  outputs = pipeline(input_src, **infer_config)
  results = [x["generated_text"] for x in outputs]

  return results[0]


def generation_config():
  with st.expander("Advanced generation config"):
    col1, col2 = st.columns(2)
    with col1:
      temperature = st.slider("Temperature", value=1., min_value=0.05, max_value=2., step=0.05)
      do_sample = st.checkbox("Do sampling", value=False)
    with col2:
      num_beams = st.slider("Beam size", value=4, min_value=1, max_value=10, step=1)
      batch_size = st.slider("Batch size", value=8, min_value=1, max_value=16, step=1)

  return dict(
    num_beams=num_beams,
    do_sample=do_sample,
    temperature=temperature,
    batch_size=batch_size,
    max_length=120,
    no_repeat_ngram_size=6,
    repetition_penalty=1.2,
  )


st.set_page_config(
  page_title="Shiki Translation",
  page_icon="🍙",
  layout="wide",
  initial_sidebar_state="expanded",
)

with st.sidebar:
  tab = option_menu(
    "🍙 Shiki Translation",
    ["Context-aware", "Raw Input", "About"],
    icons=["translate", "gear-wide-connected", "info-square"],
    menu_icon="🍙",
  )

  st.markdown("""
---
<center>🍙 Shiki Translation</center>
""", unsafe_allow_html=True,)

if tab != "Context-aware" and "CTX_TRANSLATION" in st.session_state:
  del st.session_state["CTX_TRANSLATION"]


if tab == "Context-aware":
  col1, col2 = st.columns(2)
  with col1:
    src_lang = st.selectbox("Source Language", ["auto", *langs], format_func=lambda x: format_langs[x])
  with col2:
    trg_lang = st.selectbox("Target Language", langs, format_func=lambda x: format_langs[x])

  source = st.text_area("Text to be translated")

  infer_config = generation_config()

  col1, col2 = st.columns(2)

  with col1:
    click = st.button("Translate")
  with col2:
    aligned_output = st.checkbox("Align output sentences")

  if click:
    with st.spinner("Translating..."):
      pipeline = create_pipeline()
      translate(pipeline, src_lang, trg_lang, source, infer_config)

  if "CTX_TRANSLATION" in st.session_state:
    detected_lang, sentences, results, joint_sentences = st.session_state["CTX_TRANSLATION"]

    if src_lang == "auto":
      st.info(f"Detected {format_langs[detected_lang]}")

    if aligned_output:
      for src, trg in zip(sentences, results):
        st.text(src + "\n" + trg)
    else:
      st.write(joint_sentences)


elif tab == "Raw Input":
  raw_input = st.text_area("Raw input")
  infer_config = generation_config()

  click = st.button("Execute")

  if click:
    with st.spinner("Executing..."):
      pipeline = create_pipeline()
      raw_output = translate_raw(pipeline, raw_input, infer_config)

    st.write(raw_output)


elif tab == "About":
  st.markdown("""
## Shiki Translation

This is the demo app for context-aware translation of [K024/shiki-mt5-streaming](https://huggingface.co/K024/shiki-mt5-streaming).
This model isn't aimed to handle English and may generate pool English output.

In context-aware mode, input text is splitted into sentences and translated with corresponding context.
The actual model input will be like 
```
context before. </s> closer context before. </s> en2ja: sentence to be translated. </s> context after.
```
You can test this behaviour directly in raw input mode.
""")
