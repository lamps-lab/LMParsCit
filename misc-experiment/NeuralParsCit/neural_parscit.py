#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 3, 2024

@author: muntabir choudhury
"""

# loading libraries
import pandas as pd
from collections import defaultdict
from sciwing.models.neural_parscit import NeuralParscit

# load the citation strings for evaluation
etdCite = pd.read_csv('./data-etdcite/ETDCite_annotated_data_final.csv', encoding='utf-8')
citation_strings = etdCite['CitationStrings']

# calling Neural ParsCit
neural_parscit = NeuralParscit()

tokens = [txt.split() for txt in citation_strings]
predictions = []

# generate predictions for each citation string
for txt in citation_strings:
    pred_str = neural_parscit.predict_for_text(txt)  # prediction at the token level for the citation
    pred_labels = [label.split()[-1] for label in pred_str.split()]  # extract individual labels
    predictions.append(pred_labels)

######## Debugging Purpose ###########################
# print(tokens)
# print(f"length of the tokens: {len(tokens)}")
# print(predictions)
# print(f"length of the predictions: {len(predictions)}")

# Check alignment between tokens and predictions
# for i, (tok, pred) in enumerate(zip(tokens, predictions)):
#     if len(tok) != len(pred):
#         print(f"Mismatch in entry {i}: {len(tok)} tokens vs {len(pred)} predictions")
#     else:
#         print(f"Entry {i} aligned: {len(tok)} tokens")
#########################################################

def glue_tokens_by_labels(tokens, labels):
    """grouping tokens according to their corresponding labels."""
    grouped = defaultdict(list)
    for token, label in zip(tokens, labels):
        grouped[label].append(token)
    
    # glue tokens for each label to form complete text
    glued_tokens = {label: " ".join(group) for label, group in grouped.items()}
    return glued_tokens

data = []
for tok, pred in zip(tokens, predictions):
    glued = glue_tokens_by_labels(tok, pred)  # group and glue tokens
    data.append(glued) 

result = pd.DataFrame(data).fillna('')
result.to_csv("predictions_ETDCite-NeuralParsCit.csv", encoding='utf-8', index=False)