# LMParsCit

This ongoing project leverages LLMs (e.g., llama3-8b-instruct, GPT-3.5 turbo, and GPT-4o-mini) to extract key metadata fields—like title, author, venue, and year—from references across a range of bibliography types, including journals, conference proceedings, book chapters, tech reports, and theses. It also supports multiple citation styles (e.g., IEEE, ACM, APA) and spans different academic disciplines, including STEM and non-STEM majors.

## We have used three types of techniques for this task:
-- GPT-3.5-turbo with few-shot learning (misc-experiment/GPTmodel_LMParsCit_fewshot.ipynb) leveraging prompt engineering
-- GPT-4o-mini fine-tuning using a subset of GIANT-1B synthetic dataset (misc-experiment/GPT_4o_model_LMParsCit_finetuning.ipynb)
-- Llama-3-8binstruct fine-tuning using a subset of GIANT-1B synthetic dataset (src/)
