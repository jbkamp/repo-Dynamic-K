from transformers import DistilBertTokenizer
import pandas as pd
import re
import math
import numpy as np
from IPython.core.display import HTML
import webbrowser
import random

df_train_1 = pd.read_csv("./esnli/data_original/esnli_train_1.csv")
df_train_2 = pd.read_csv("./esnli/data_original/esnli_train_2.csv")
df_train = pd.concat([df_train_1,df_train_2])
df_dev = pd.read_csv("./esnli/data_original/esnli_dev.csv")
df_test = pd.read_csv("./esnli/data_original/esnli_test.csv")

## Some (6) examples in the training set miss a "Sentence2" -> remove
print(len(df_train[df_train['Sentence1'].notna()])) #549367 vvv should be equal
print(len(df_train[df_train['Sentence2'].notna()])) #549361 ^^^ should be equal
df_train = df_train[df_train['Sentence2'].notna()] # remove 6 lines where 'Sentence2' missing
print(len(df_train[df_train['Sentence1'].notna()])) #549361 vvv are now equal
print(len(df_train[df_train['Sentence2'].notna()])) #549361 ^^^ are now equal
##
print(len(df_dev[df_dev['Sentence1'].notna()])) #9842 vvv are already equal
print(len(df_dev[df_dev['Sentence2'].notna()])) #9842 ^^^ are already equal
##
print(len(df_test[df_test['Sentence1'].notna()])) #9824 vvv are already equal
print(len(df_test[df_test['Sentence2'].notna()])) #9824 ^^^ are already equal

# stats on rationales
df_train["split"] = "train"
df_dev["split"] = "dev"
df_test["split"] = "test"
df_devtest = pd.concat([df_dev,df_test])

df_devtest = df_devtest.drop(columns=['Sentence1_Highlighted_1',
                              'Sentence1_Highlighted_2',
                              'Sentence1_Highlighted_3',
                              'Sentence2_Highlighted_1',
                              'Sentence2_Highlighted_2',
                              'Sentence2_Highlighted_3',
                              'Explanation_1',
                              'Explanation_2',
                              'Explanation_3'])

'''
train has the columns 
    `Sentence1_marked_1, Sentence2_marked_1`
dev and test have the columns
    `Sentence1_marked_1, Sentence2_marked_1`
    `Sentence1_marked_2, Sentence2_marked_2`
    `Sentence1_marked_3, Sentence2_marked_3` 
    `Explanation_1, Explanation_2, Explanation_3`
'''

## we want:
#   check that each example has minimum 3 tokens
#   min, avg, stdev, max highlighted tokens per premise
#   min, avg, stdev, max highlighted tokens per hypothesis
#   min, avg, stdev, max highlighted tokens per premise+hypothesis
#   inter-annotator agreement over tokens

def count_nx(sentence):
    """
    :param sentence: e.g. "hello this *is* a *very* *8portant* meeting *alright?*"
    :return: count of tokens between asterisks, e.g. 4
    """
    return len(re.findall("\*[a-zA-Z0-9_]",sentence))

def create_count_col(sentence_col):
    """
    :param sentence_col: column of dataframe that contains a sentence to apply `count_nx` to
    :return: new column (list) containing the counts for each sentence
    """
    count_col = []
    for sentence in sentence_col:
        if isinstance(sentence,str):
            count_col.append(count_nx(sentence))
        else:
            print(type(sentence), sentence)
            count_col.append(math.nan)
    return count_col

# inspecting df_train
df_train["Sentence1_nx"] = create_count_col(df_train["Sentence1_marked_1"])
df_train["Sentence2_nx"] = create_count_col(df_train["Sentence2_marked_1"])
df_train["Sentence12_nx"] = df_train["Sentence1_nx"] + df_train["Sentence2_nx"]
np.mean(df_train["Sentence1_nx"]) # 1.817
np.mean(df_train["Sentence2_nx"]) # 2.196
np.mean(df_train["Sentence12_nx"]) # 4.013 <---
np.std(df_train["Sentence1_nx"]) # 2.091
np.std(df_train["Sentence2_nx"]) # 1.542
np.std(df_train["Sentence12_nx"]) # 3.007 <---
np.min(df_train["Sentence1_nx"]) # 0
np.min(df_train["Sentence2_nx"]) # 0
np.min(df_train["Sentence12_nx"]) # 0 <---
np.max(df_train["Sentence1_nx"]) # 39
np.max(df_train["Sentence2_nx"]) # 39
np.max(df_train["Sentence12_nx"]) # 78 <---
len(df_train[df_train["Sentence1_nx"]==0]) #182820
len(df_train[df_train["Sentence2_nx"]==0]) #3981
len(df_train[df_train["Sentence12_nx"]==0]) #52 <---
len(df_train[df_train["Sentence12_nx"]<3]) #226967 <---

# inspecting df_devtest
'''
count only Sentence12 for each example, otherwise too many variable (and not that relevant since we already have stats on the training set rationales)
'''
df_devtest["Sentence12_nx_1"] = [l1+l2 for l1,l2 in zip(create_count_col(df_devtest["Sentence1_marked_1"]), #annotator 1
                                                        create_count_col(df_devtest["Sentence2_marked_1"]))]
df_devtest["Sentence12_nx_2"] = [l1+l2 for l1,l2 in zip(create_count_col(df_devtest["Sentence1_marked_2"]), #annotator 2
                                                        create_count_col(df_devtest["Sentence2_marked_2"]))]
df_devtest["Sentence12_nx_3"] = [l1+l2 for l1,l2 in zip(create_count_col(df_devtest["Sentence1_marked_3"]), #annotator 3
                                                        create_count_col(df_devtest["Sentence2_marked_3"]))]

np.mean(df_devtest["Sentence12_nx_1"]) # 4.289          #annotator 1
np.mean(df_devtest["Sentence12_nx_2"]) # 4.437          #annotator  2
np.mean(df_devtest["Sentence12_nx_3"]) # 4.324          #annotator      3
np.std(df_devtest["Sentence12_nx_1"]) # 3.137           #annotator 1
np.std(df_devtest["Sentence12_nx_2"]) # 3.116           #annotator  2
np.std(df_devtest["Sentence12_nx_3"]) # 3.133           #annotator      3
np.min(df_devtest["Sentence12_nx_1"]) # 0               #annotator 1
np.min(df_devtest["Sentence12_nx_2"]) # 1               #annotator  2
np.min(df_devtest["Sentence12_nx_3"]) # 0               #annotator      3
np.max(df_devtest["Sentence12_nx_1"]) # 42              #annotator 1
np.max(df_devtest["Sentence12_nx_2"]) # 46              #annotator  2
np.max(df_devtest["Sentence12_nx_3"]) # 32              #annotator      3

ranges = [np.max(n1n2n3) - np.min(n1n2n3) for n1n2n3 in zip(df_devtest["Sentence12_nx_1"],
                                                            df_devtest["Sentence12_nx_2"],
                                                            df_devtest["Sentence12_nx_3"])]
np.mean(ranges) # 3.464

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def spanmask(sentence):
    """
    :param sentence: e.g.                               "hello this *is* a *very* *8portant* meeting *alright?*"
    :return: spanmask of tokens between asterisks, e.g. [0,    0,    1,  0,1,     1,         0,      1       0 ]
        !! punctuation is always given 0, since Camburu's annotators could not differentiate between punct and non-p
    """
    tokenized_sentence = tokenizer.tokenize(sentence)
    tmp_mask = []
    asterisk_pair_counter = -1
    missing_pair_flag = -1
    for token in tokenized_sentence:
        if token == "*":
            tmp_mask.append(asterisk_pair_counter)
            missing_pair_flag = -missing_pair_flag
            if missing_pair_flag == -1:
                asterisk_pair_counter += -2
        elif re.match(r'^[^\w\s]+$',token):
            tmp_mask.append(0)
        else:
            tmp_mask.append(2)
    mask = []
    lookup = dict(enumerate(tmp_mask))
    for i, value in lookup.items():
        if value == 0:
            mask.append(0)
        elif value == 2:
            if sum([v for v in list(lookup.values())[:i] if v < 0]) % 2 == 0:
                mask.append(0)
            else:
                mask.append(1)
    #return tokenized_sentence, tmp_mask, mask
    return mask

def create_spanmask_col(sentence_col):
    """
    :param sentence_col: column of dataframe that contains a sentence to compute `spanmask` for
    :return: new column (list) containing the span mask for each sentence
    """
    spanmask_col = []
    for sentence in sentence_col:
        if isinstance(sentence,str):
            spanmask_col.append(spanmask(sentence))
        else:
            print(type(sentence), sentence)
            spanmask_col.append(math.nan)
    return spanmask_col

df_devtest["Sentence12_spanmask_1"] = [l1+l2 for l1,l2 in zip(create_spanmask_col(df_devtest["Sentence1_marked_1"]), #annotator 1
                                                              create_spanmask_col(df_devtest["Sentence2_marked_1"]))]
df_devtest["Sentence12_spanmask_2"] = [l1+l2 for l1,l2 in zip(create_spanmask_col(df_devtest["Sentence1_marked_2"]), #annotator 2
                                                              create_spanmask_col(df_devtest["Sentence2_marked_2"]))]
df_devtest["Sentence12_spanmask_3"] = [l1+l2 for l1,l2 in zip(create_spanmask_col(df_devtest["Sentence1_marked_3"]), #annotator 3
                                                              create_spanmask_col(df_devtest["Sentence2_marked_3"]))]

def iou(spanmask1,spanmask2,spanmask3):
    """
    :param spanmask1: binary list
    :param spanmask2: binary list
    :param spanmask3: binary list
    :return: intersection over union, based on the same-indexed elements of the three lists
    """
    assert len(spanmask1) == len(spanmask2)
    assert len(spanmask2) == len(spanmask3)

    intersection_spanmask = []
    for i in range(len(spanmask1)):
        if spanmask1[i] == 1 and spanmask2[i] == 1 and spanmask3[i] == 1:
            intersection_spanmask.append(1)
        else:
            intersection_spanmask.append(0)
    union_spanmask = []
    for i in range(len(spanmask1)):
        if spanmask1[i] == 0 and spanmask2[i] == 0 and spanmask3[i] == 0:
            union_spanmask.append(0)
        else:
            union_spanmask.append(1)

    intersection_over_union = np.sum(intersection_spanmask) / np.sum(union_spanmask)
    intersection_over_min = np.sum(intersection_spanmask) / min(np.sum(spanmask1),np.sum(spanmask2),np.sum(spanmask3))
    return intersection_over_union, intersection_over_min

iou([1,1,1],[1,0,1],[0,0,1]) #test works: 0.33 en 1.0

df_devtest["iou"] = [iou(l1,l2,l3)[0] for l1,l2,l3 in zip(df_devtest["Sentence12_spanmask_1"],
                                                       df_devtest["Sentence12_spanmask_2"],
                                                       df_devtest["Sentence12_spanmask_3"])]
df_devtest["iom"] = [iou(l1,l2,l3)[1] for l1,l2,l3 in zip(df_devtest["Sentence12_spanmask_1"],
                                                       df_devtest["Sentence12_spanmask_2"],
                                                       df_devtest["Sentence12_spanmask_3"])]
np.mean(df_devtest["iou"]) # 0.35
np.mean(df_devtest["iom"]) # 0.72

#######

df_dev = df_devtest[df_devtest["split"] == "dev"]

def get_human_aggreg_values(spanmask1,spanmask2,spanmask3):
    """
    :param spanmask1:   [1,0,1,1,1]
    :param spanmask2:   [1,0,0,0,1]
    :param spanmask3:   [1,0,1,0,1]
    :return:            [3,0,2,1,3]
    """
    aggreg_mask = [sum(tup) for tup in zip(spanmask1,spanmask2,spanmask3)]
    return aggreg_mask

df_dev["human_aggreg_values"] = [get_human_aggreg_values(l1,l2,l3) for l1,l2,l3 in zip(df_dev["Sentence12_spanmask_1"],
                                                       df_dev["Sentence12_spanmask_2"],
                                                       df_dev["Sentence12_spanmask_3"])]
df_dev["tokens"] = df_dev["Sentence1"] .str.cat(df_dev["Sentence2"], sep=' ').str.split(' ')
df_dev["token_aggreg_tuples"] = df_dev.apply(lambda row: list(zip(row['tokens'], row['human_aggreg_values'])), axis=1)

"""
color-code attributions
- human overlap (1x 1line per example, with gradients of white) --> `esnli_dataset_and_annotations_stats`
--- different color codes for scores 0,1,2,3
"""

def colorcode_strings_human_agreement(lst):
    # Define CSS styles
    css = """
    <style>
    .container {
        display: flex;
        flex-direction: row;
    }
    table {
        border-collapse: collapse;
        margin-right: 20px;
    }
    td {
        padding: 5px;
        text-align: center;
        font-family: sans-serif;
    }
    .bg-black {
        background-color: black;
        color: white;
    }
    .bg-grey-1 {
        background-color: #555555;
        color: white;
    }
    .bg-grey-2 {
        background-color: #999999;
        color: white;
    }
    .bg-white {
        background-color: white;
        color: black;
    }
    .legend-table {
        margin-left: 20px;
    }
    </style>
    """
    # Create table header
    header = "<tr>" + "".join([f"<td>{i + 1}</td>" for i in range(len(lst))]) + "</tr>"
    # Create table rows
    rows = "<tr>" + "".join([f"<td class='bg-black'>{t[0]}</td>" if t[1] == 0 else \
                             f"<td class='bg-grey-1'>{t[0]}</td>" if t[1] == 1 else \
                             f"<td class='bg-grey-2'>{t[0]}</td>" if t[1] == 2 else \
                             f"<td class='bg-white'>{t[0]}</td>" for t in lst]) + "</tr>"
    # Combine header and rows into HTML table
    table = f"<table>{header}{rows}</table>"
    # Create legend table
    legend = "<table class='legend-table'>"
    legend += "<tr><th>Color Code</th><th>Index</th></tr>"
    legend += "<tr><td class='bg-black'></td><td>0</td></tr>"
    legend += "<tr><td class='bg-grey-1'></td><td>1</td></tr>"
    legend += "<tr><td class='bg-grey-2'></td><td>2</td></tr>"
    legend += "<tr><td class='bg-white'></td><td>3</td></tr>"
    legend += "</table>"
    # Combine table and legend into a container div
    container = f"<div class='container'>{table}{legend}</div>"
    # Combine CSS and container into final HTML output
    output = f"{css}{container}"
    # Return HTML object
    return HTML(output)

#generate and visualize one html file with 20 colorcoded examples, randomly sampled from dev set
html_list = [colorcode_strings_human_agreement(df_dev["token_aggreg_tuples"][i]) for i in random.sample(range(len(df_dev)),20)]  #20 html objects
html_string = "<html><body>"+("<p></p>".join(html.data for html in html_list)) + "</body></html>" #concat html objects
with open("./esnli/explanations/agreement_human_viz.html", "w") as file: #save to file
    file.write(html_string)

# open file in browser; CHANGE ABS PATH: <YOUR PATH>
webbrowser.open_new_tab('file:///<YOUR PATH>/esnli/explanations/agreement_human_viz.html') #open in browser
