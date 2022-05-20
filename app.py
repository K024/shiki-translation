import os
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration, Text2TextGenerationPipeline
import gradio as gr
import text_utils

langs = ["zh", "ja", "en"]

model_config = dict(
  pretrained_model_name_or_path="K024/shiki-mt5-streaming",
  revision="main",
  use_auth_token=os.environ["ACCESS_TOKEN"],
)

pipeline = Text2TextGenerationPipeline(
  model=MT5ForConditionalGeneration.from_pretrained(**model_config),
  tokenizer=T5Tokenizer.from_pretrained(**model_config),
  device=0 if torch.cuda.is_available() else -1,
)

def swap_lang(source, target):
  if source == "auto":
    return [target, "zh"]
  return [target, source]


def translate(src_lang, trg_lang, source):
  if not source:
    return [src_lang, ""]

  if src_lang == "auto":
    src_lang = text_utils.detect_lang(source)

  if src_lang not in langs:
    return ["auto", f"Error: detected language '{src_lang}' is not supported."]

  if src_lang == trg_lang:
    return [src_lang, "Error: detected language is same as the target language."]

  sentences, endls = text_utils.split_sentences(source)
  input_src = text_utils.prepare_contexts(sentences, src_lang, trg_lang)

  infer_config = dict(
    no_repeat_ngram_size=6,
    repetition_penalty=1.2,
    max_length=120,
    do_sample=False,
    num_beams=4,
    batch_size=4,
  )

  outputs = pipeline(input_src, **infer_config)
  results = [x["generated_text"] for x in outputs]
  joint_sentences = text_utils.join_sentences(results, endls, trg_lang)

  return [src_lang, joint_sentences]


def translate_raw(source):

  infer_config = dict(
    no_repeat_ngram_size=6,
    repetition_penalty=1.2,
    max_length=120,
    do_sample=False,
    num_beams=4,
    batch_size=4,
  )

  input_src = [source]
  outputs = pipeline(input_src, **infer_config)
  results = [x["generated_text"] for x in outputs]

  return results[0]


demo = gr.Blocks()
with demo:
  gr.Markdown("""
## Shiki Translation Demo

This is the demo code for context-aware translation of [K024/shiki-mt5-streaming](https://huggingface.co/K024/shiki-mt5-streaming).
This model isn't aimed to handle English and may generate pool English output.  

In context-aware mode, input text is splitted into sentences and translated with corresponding context.
The actual model input will be like `context before. </s> closer context before. </s> en2ja: sentence to be translated. </s> context after.`.
You can test this behaviour directly in raw input mode.

""")

  with gr.Tabs():
    with gr.TabItem("Context-Aware Translation"):
      with gr.Row():
        source_lang = gr.Dropdown(["auto"] + langs, label="Source Language")
        target_lang = gr.Dropdown(langs, label="Target Language")
        swap_btn = gr.Button("Swap", variant="secondary")
        swap_btn.click(swap_lang, [source_lang, target_lang], [source_lang, target_lang])
      with gr.Row():
        input_text = gr.Textbox(lines=7, label="Input", placeholder="Text to be translated...")
        output_text = gr.Textbox(lines=7, label="Translated", placeholder="Translated text...", interactive=False)
      translate_btn = gr.Button("Translate")
      translate_btn.click(translate, [source_lang, target_lang, input_text], [source_lang, output_text])

    with gr.TabItem("Raw Input Mode"):
      with gr.Row():
        input_text_raw = gr.Textbox(lines=7, label="Raw Input", placeholder="Raw input text...")
        output_text_raw = gr.Textbox(lines=7, label="Output", placeholder="Output text...", interactive=False)
      translate_btn_raw = gr.Button("Execute")
      translate_btn_raw.click(translate_raw, [input_text_raw], [output_text_raw])

if __name__ == "__main__":
  demo.launch()
