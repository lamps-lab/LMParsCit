#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 01:52:15 2024

@author: muntabir choudhury
"""


import pandas as pd
import re
from datasets import Dataset


# Function to extract text between tags and remove subtags
def extract_text(tag, text):
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text)
    if match:
        # Remove subtags
        return re.sub(r'<.*?>', '', match.group(1)).strip()
    else:
        return None

# Read citation strings from text file
with open('./../data-giant/10K_citation.txt', 'r') as file:
    citation_strings = file.read()

# Extract metadata fields
metadata = ['author', 'title', 'issued', 'container-title', 'editor translator', 'publisher']
data = []
for citation_string in citation_strings.split('\n'):
    fields = {}
    for field in metadata:
        fields[field] = extract_text(field, citation_string)
    fields = {k: v if pd.notna(v) else '' for k, v in fields.items()}
    fields['text'] = f"### system:\n\nYou are an advanced Citation Parsing system designed for extracting metadata from citation strings. Your task is to identify and extract specific metadata fields from various citation styles, including IEEE, ACM, APA, MLA, and Chicago. The metadata fields are defined as follows: a) Title: The title of the paper or article. b) Author: The names of the authors involved in the publication. It should be a 'person' name. c) Container-Title: The name of the publication venue, such as a journal name, conference proceedings, book title, thesis, or technical report. For example, if the cited work is an article, the 'container-title' would be the journal name or the name of the conference proceedings where the article was published. d) Issued: The date of publication, including year, month, and day if available. Do not get confused this field with the volume number. Ignore the volume number (i.e., it is a part of a hierarchical system used to organize and identify issues within a publication) and page number. e) Editor: The names of the editors who contributed to a book, journal, or other publication. This field include terms like 'Eds.', 'Editors', '(Eds.)', 'Editor', or '(Editors)' after the end of the person names and sometimes precede with 'In'. Do not get confused with the author names. The author names do not include the terms like 'Eds.', 'Editors', '(Eds.)', 'Editor', or '(Editors)' after the end of the names. Publisher: the organization or company responsible for producing and distributing the publication. Your goal is to accurately extract these fields from the provided citation strings. ### user:\n\n{', '.join(fields.values())}\n\n### assistant:\n\ntitle: {fields['title']}\n\nauthor: {fields['author']}\n\nissued: {fields['issued']}\n\ncontainer-title: {fields['container-title']}\n\neditor: {fields['editor translator']}\n\npublisher: {fields['publisher']}"
    data.append(fields)


# Create DataFrame
df = pd.DataFrame(data)

# Add full citation string column
df['citations'] = citation_strings.split('\n')

# Reorder columns
df = df[['citations'] + metadata + ['text']]

# Define the reformatting function (applying special tokens)
def reformat_text(text):
    # Split the text by "### Output:"
    parts = text.split("### assistant:")

    if len(parts) == 2:
        sys_instruction = parts[0].split("### user:")[0].replace("### system:", "").strip()
        instruction = parts[0].split("### user:")[1].strip()
        output = parts[1].strip()

        # Format the instruction and output into Meta Llama-3 Instruct format
        # the <|eot_id|> indicates the end of the text and helps llama from repeating tokens
        formatted_output = f"<|begin_of_text|><|start_header_id|>system:<|end_header_id|> {sys_instruction}<|eot_id|><|start_header_id|>\nuser:<|end_header_id|> {instruction}<|end_of_text|><|eot_id|><|start_header_id|>\nassistant:<|end_header_id|> {output}<|end_of_text|><|eot_id|>"
        return formatted_output
    else:
        return text  # Return the original text if the split didn't work correctly

# Apply the reformatting function to the 'text' column
df['text'] = df['text'].apply(reformat_text)

# Save DataFrame to CSV
df.to_csv('./../data-giant/input/citation_data-GIANT10K-llama3.csv', index=False)

#print(df)


# Convert the DataFrame to a dictionary format
data_dict = {
    'citations': df['citations'].tolist(),
    'author': df['author'].tolist(),
    'title': df['title'].tolist(),
    'date': df['issued'].tolist(),
    'container-title': df['container-title'].tolist(),
    'publisher': df['publisher'].tolist(),
    'text': df['text'].tolist(),
}

# Display the dictionary
#print(data_dict)

# Assuming data_dict is already defined
dataset = Dataset.from_dict(data_dict)
print(dataset)

