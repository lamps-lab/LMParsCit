# LMParsCit

## Leveraging different large language models (llama3-8b-instruct, GPT3.5 turbo, GPT4o-mini), this application will extract key metadata fields (e.g., title, author, venue, and yearr) from references in various biblipography types (e.g., journal, conference proceedings, book, in-book, technical report, and thesis), and various citation styles (e.g., IEEE, ACM, Chicago, MLA, etc.) from different academic disciplines (e.g., Arts and Humanities, Education, Computer Science, Engineering, etc.).

## We have used three types of techniques for this task:
-- GPT-3.5-turbo with fewshot learning (misc-experiment/GPTmodel_LMParsCit_fewshot.ipynb) leveraging prompt engineering
-- GPT-4o-mini fine-tuning using a subset of GIANT-1B synthetic dataset (misc-experiment/GPT_4o_model_LMParsCit_finetuning.ipynb)
-- Llama-3-8binstruct fine-tuning using a subset of GIANT-1B synthetic dataset (src/)