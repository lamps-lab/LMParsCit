#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 02:56:15 2024

@author: muntabir
"""
import pandas as pd
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

"""### Format Test set (e.g., CORA)"""
# Load the CSV file
file_path = './../cora-ref/CORA-for-eval.csv'
df = pd.read_csv(file_path)

# Reformat the 'CitationString' column
df['text'] = df['CitationString'].apply(lambda x:f"<|start_header_id|>system:<|end_header_id|> You are an advanced Citation Parsing system designed for extracting metadata from citation strings. Your task is to identify and extract specific metadata fields from various citation styles, including IEEE, ACM, APA, MLA, and Chicago. The metadata fields are defined as follows: a) Title: The title of the paper or article. b) Author: The names of the authors involved in the publication. It should be a 'person' name. c) Container-Title: The name of the publication venue, such as a journal name, conference proceedings, book title, thesis, or technical report. For example, if the cited work is an article, the 'container-title' would be the journal name or the name of the conference proceedings where the article was published. d) Issued: The date of publication, including year, month, and day if available. Do not get confused this field with the volume number. Ignore the volume number (i.e., it is a part of a hierarchical system used to organize and identify issues within a publication) and page number. e) Editor: The names of the editors who contributed to a book, journal, or other publication. This field include terms like 'Eds.', 'Editors', '(Eds.)', 'Editor', or '(Editors)' after the end of the person names and sometimes precede with 'In'. Do not get confused with the author names. The author names do not include the terms like 'Eds.', 'Editors', '(Eds.)', 'Editor', or '(Editors)' after the end of the names. Publisher: the organization or company responsible for producing and distributing the publication. Your goal is to accurately extract these fields from the provided citation strings.<|eot_id|><|start_header_id|>user:<|end_header_id|> {x}<|end_of_text|><|eot_id|>")

# Save the modified DataFrame to a new CSV file
output_path = './../cora-ref/CORA-eval-reformat-llama3.csv'
df.to_csv(output_path, index=False)

print(f"CSV file saved to {output_path}")

testCora = pd.read_csv("./../cora-ref/CORA-eval-reformat-llama3.csv")

testCora.drop(columns=['volume'], inplace=True)
print(testCora)


model_name = "./../llama3-lmparsCit-model/llama-3-8b-Instruct-bnb-4bit-lmparscit-10K"
max_seq_length = 1056
dtype = None
load_in_4bit = True
seed = 3407

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Using FastLanguageModel for fast inference
FastLanguageModel.for_inference(model)

predictions = []
# Tokenizing the input and generating the output
for i in range(len(testCora)):
    inputs = tokenizer(
        [
            testCora['text'][i]
        ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 400, use_cache = True)
    predictions.append(tokenizer.batch_decode(outputs, skip_special_tokens = False))

print("****** Prediction is Complete!!*********")

testDF = pd.DataFrame(predictions, columns = ['pred_text'])
testDF.to_csv("./../pred_output/predictions_CORA-llama3-8b.csv", index=False)