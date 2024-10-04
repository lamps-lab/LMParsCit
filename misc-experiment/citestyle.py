#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:06:42 2024

@author: muntabir
"""

import pandas as pd
from styleclass.classify import classify
from styleclass.train import get_default_model

data = pd.read_csv("bibliography_types.csv")
data_citation = data['CitationStrings']

#lst = []
#for strings in data_citation:
#    quoteString = "'"+strings+"'"

model = get_default_model()

stylelist = []
for strings in data['CitationStrings']:
    prediction = classify(strings, *model)
    stylelist.append(prediction)

cite_style = pd.DataFrame(stylelist, columns = ["CitationStyle"])
concat = pd.concat([data, cite_style], axis = 1)
concat.to_csv('bibliography_style.csv', index = None)