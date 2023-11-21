from transformers import DistilBertTokenizer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import random
import pickle
from itertools import product
from IPython.core.display import HTML
import re
import webbrowser
from collections import Counter

"""
Load tokenizer and pickled explanations after model selection with lowest APD (run 08)
"""
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
with open("./explanations/test_explanations/test_dataset_explanations_db_08.pickle", "rb") as file:
    test_dataset_explanations = pickle.load(file)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
"""
Main functions for computing relevance and agreement
"""
def create_topk_mask(attribs, tokens, topk=-1, ranked=False, rm_special_tokens=True, dynamic=False):
    """
    :param attribs:             list of token attributions
    :param tokens:              list of tokens where the attribs were computed for
    :param topk:                defaults to -1. If dynamic==False, topk should be set to a positive integer
    :param ranked:              boolean
    :param rm_special_tokens:   defaults to True
    :param dynamic:             defaults to False; set to True if mask is to be computed for dyn. top-k based on loc max
    :return:                    boolean mask based on `attribs` list, with 1s indexed at topk highest attribution values
        e.g. create_topk_mask([1,2,3,3,4,5],["a","b","c","d","[CLS]","[SEP]"],3) returns [0, 1, 1, 1]
        e.g. create_topk_mask([1,2,3,3,4,5],["a","b","c","d","[CLS]","[SEP]"],3,rm_special_tokens=False)
        --> returns [0, 0, 1, 0, 1, 1] (or [0, 0, 0, 1, 1, 1], because ties are solved by random choice)
    """
    assert len(attribs) == len(tokens)
    if dynamic == False: #regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0

    if rm_special_tokens:
        attribs = [a for t,a in zip(tokens, attribs) if t not in {"[CLS]","[SEP]","<s>","</s>","Ġ"}]
    if dynamic:
        assert topk == -1 #contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
        attribs_2, local_maxima_indices = compute_and_plot_local_maxima(list_of_floats=attribs, plot=False)
        dynamic_topk_mask = [0 for _ in attribs_2]
        for loc_max_i in local_maxima_indices:
            dynamic_topk_mask[loc_max_i] = 1
        return dynamic_topk_mask

    assign_indices = list(enumerate(attribs))
    assign_indices.sort(key=lambda tup: (tup[1], random.random())) #if tie -> randomize
    sorted_indices = [i for i,a in assign_indices]
    topk_sorted_indices = sorted_indices[-topk:] #e.g. top-2 of [1,2,3,0] is [3,0]
    topk_mask = [0 for a in attribs] #initialize 0s mask
    if ranked:
        for i, rank in zip(topk_sorted_indices,range(len(topk_sorted_indices),0,-1)):
            topk_mask[i] = rank #assign rank at topk indices; outside the topk remains 0
    else:
        for i in topk_sorted_indices:
            topk_mask[i] = 1 #assign 1 at topk indices; outside the topk remains 0
    return topk_mask

def get_token_agr_score(tup, class_1_agreement=True):
    """
    :param tup: tuple of top-k mask for a specific token; unranked or ranked does not matter
    :param class_1_agreement: only calculate for class 1
    :return: agreement ratio for a token
        e.g. get_token_agr_score((0,0,0,1,0))                           -> 0.8
        e.g. get_token_agr_score((0,0,0,1,0), class_1_agreement=True)   -> 0.2
    """
    if sum(tup) == 0: # e.g. vertical (0,0,0,0,0)
        agr = np.nan  # we assign nan and not 100% agreement if all methods assign 0
    else:
        d = {rank:0 for rank in set(tup)} # initialize, e.g. {0:0, 1:0, 2:0, 3:0} or {0:0, 1:0}
        for rank in tup: #abs freq of ranks; e.g. {0:2, 1:1, 2:1, 3:1}
            d[rank] += 1
        for rank, freq in d.items(): #relative freq of ranks; e.g. {0:0.4, 1:0.2, 2:0.2, 3:0.2}
            d[rank] = freq/len(tup)
        if class_1_agreement:
            agr = max([rel_freq for (rank, rel_freq) in d.items()
                       if rank != 0])  # agreement is the max of relative freqs for ranks != 0; e.g. 0.2
        else:
            agr = max(d.values())  # agreement is the max of relative freqs; e.g. 0.4

    return agr

def get_instance_agr_score(explanations, topk=-1, ranked=False, rm_special_tokens=True, class_1_agreement=True,
                           leaveout_idx_list=[], human_aggreg_values=[], dynamic=False):
    """
    :param explanations:        ferret output object
    :param topk:                top-k parameter
    :param ranked:
    :param rm_special_tokens:
    :param class_1_agreement:
    :param leaveout_idx_list:   list containing 0 or + integers in the range [0,5] to excl. from agreement computation
        0 ~ Partition SHAP
        1 ~ LIME
        2 ~ Gradient
        3 ~ Gradient (xInput)
        4 ~ Integrated Gradient
        5 ~ Integrated Gradient (xInput)
    :param human_aggreg_values: list containing human aggregation values (obtained from annotated gold highlights)
    :return:                    mean agreement score over tokens for the instance, based on top-k and ignoring
                                nan values. If topk provided is <1 or >n_tokens in the sentence,
                                return instance_agr_score=nan
    """
    if dynamic == False: #regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0
    else:
        assert topk == -1 #contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
    # leave one or more attribution methods out depending on parameter
    explanations = [x for n,x in enumerate(explanations) if n not in leaveout_idx_list]
    # compute list of method-wise masks, e.g. [ [0,0,0,1,1], [0,0,1,1,1], [0,0,1,0,1] ]

    list_of_masks = [create_topk_mask(attribs=list(x.scores), #TODO rob tokenizer [s for s,t in zip(x.scores,x.tokens) if t != 'Ġ']
                                      tokens=list(x.tokens), #TODO rob tokenizer [t for s,t in zip(x.scores,x.tokens) if t != 'Ġ']
                                      topk=topk,
                                      ranked=ranked,
                                      rm_special_tokens=rm_special_tokens,
                                      dynamic=dynamic)
                     for x in explanations]
    ####################################################################################################################
    # ADD HUMAN TOP K MASK AS A ROW TO `list_of_masks` if it is added as an argument
    if len(human_aggreg_values) > 0:
        assert rm_special_tokens == True  # human gt does (can)not have special tokens
        assert ranked == False  # did not test for ranked
        assert len(human_aggreg_values) == len(list_of_masks[0])

        if not dynamic:
            human_assign_indices = list(enumerate(human_aggreg_values))
            human_assign_indices.sort(key=lambda tup: (tup[1], random.random()))  # if tie -> randomize
            human_sorted_indices = [i for i, a in human_assign_indices]
            human_topk_sorted_indices = human_sorted_indices[-topk:]  # e.g. top-2 of [1,2,3,0] is [3,0]
            human_topk_mask = [0 for _ in human_aggreg_values]  # initialize 0s mask
            for i in human_topk_sorted_indices:
                human_topk_mask[i] = 1  # assign 1 at topk indices; outside the topk remains 0
            list_of_masks.append(human_topk_mask)
        else:
            human_topk_mask = create_topk_mask(attribs=human_aggreg_values,
                                               tokens=["dummy" for _ in human_aggreg_values],
                                               topk=topk,
                                               ranked=ranked,
                                               rm_special_tokens=rm_special_tokens,
                                               dynamic=dynamic)
            list_of_masks.append(human_topk_mask)
    ####################################################################################################################
    tupled_tokenwise = list(zip(*list_of_masks))
    agreements_tokenwise = [get_token_agr_score(tup,class_1_agreement=class_1_agreement)
                            for tup in tupled_tokenwise]
    if rm_special_tokens:
        tokens = [t for t in explanations[0].tokens if t not in {"[CLS]","[SEP]","<s>","</s>"}]
    else:
        tokens = list(explanations[0].tokens)

    return (np.nanmean(agreements_tokenwise), #ignore nans
            list_of_masks,
            agreements_tokenwise,
            tokens)

def get_dataset_agr_scores(dataset_explanations, topk=-1, ranked=False, rm_special_tokens=True,
                           class_1_agreement=True, leaveout_idx_list=[], dataset_human_aggreg_values=[], dynamic=False):
    """
    :param dataset_explanations:
    :param topk:
    :param ranked:
    :param rm_special_tokens:
    :param class_1_agreement:
    :param leaveout_idx_list:           list containing 0 or + integers in the range [0,5] to excl. from agreement comp.
        0 ~ Partition SHAP
        1 ~ LIME
        2 ~ Gradient
        3 ~ Gradient (xInput)
        4 ~ Integrated Gradient
        5 ~ Integrated Gradient (xInput)
    :param dataset_human_aggreg_values:
    :param dynamic:
    :return:                            list of agreement scores for each instance of the data for a specif. top-k value
    """
    if dynamic == False:  # regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0
    else:
        assert topk == -1  # contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)

    if len(dataset_human_aggreg_values) > 0:
        assert rm_special_tokens == True #human gt does (can)not have special tokens
        assert ranked == False #did not test for ranked
        assert len(dataset_explanations) == len(dataset_human_aggreg_values)

    dataset_instance_agr_scores = []
    for i, instance_explanations in enumerate(dataset_explanations):
        if len(dataset_human_aggreg_values) > 0:
            instance_human_aggreg_values = dataset_human_aggreg_values[i]
        else:
            instance_human_aggreg_values = []

        inst_agr_score = get_instance_agr_score(explanations=instance_explanations,
                                                topk=topk,
                                                ranked=ranked,
                                                rm_special_tokens=rm_special_tokens,
                                                class_1_agreement=class_1_agreement,
                                                leaveout_idx_list=leaveout_idx_list,
                                                human_aggreg_values=instance_human_aggreg_values,
                                                dynamic=dynamic)
        mean_instance_agr_score, _, __, ___ = inst_agr_score
        dataset_instance_agr_scores.append(mean_instance_agr_score)

    return dataset_instance_agr_scores

"""
Exploration (done on dev set)
- Plot frequency distribution sentence lengths
"""
def plot_freq_distr_sent_lens():
    sent_lens = [len(x[0].tokens)-3 for x in test_dataset_explanations] # minus [CLS], [SEP], [SEP]
    counter = Counter(sent_lens)
    sentence_length = list(counter.keys())
    frequencies = list(counter.values())
    plt.bar(sentence_length, frequencies)
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')
    plt.show()

plot_freq_distr_sent_lens()

"""
Exploration (done on dev set)
- Compute dataset agreement scores
- Plot mean agreements over all dataset instances, for each topk in wide range_topks (defined below)
"""
range_topks = 40
dataset_unranked_dict = dict()

for topk in range(1, range_topks+1):
    dataset_unranked_dict[topk] = get_dataset_agr_scores(dataset_explanations=test_dataset_explanations,
                                                         topk=topk, ranked=False, rm_special_tokens=True)
    print(topk,"done")

"""
Exploration (done on dev set):
- Visualize in colorcoded html some examples of machine agreement, and save to file. 
- Examples should open in browser.
[[For colorcoded human agreement on Camburu's dataset, see `esnli_dataset_and_annotations_stats.py`]]
"""
def colorcode_strings_machine_agreement(lists, float_row, mean_instance_agr_score):
    """
    :param lists: list of (6) tupled agreement masks (token, mask) for a single instance
    :param float_row: list of token-wise agreements for a single instance
    :param mean_floats: list of mean agreement scores for each instance
    :param instance_mean_agreement: float, mean agreement score for the current instance
    :return: html object containing colorcoded strings in table format:
        -- background white if token in topk else black
        -- agr scores in tones of green: darker if float higher
    """
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
    .bg-white {
        background-color: white;
        color: black;
    }
    .bg-green-0 {
        background-color: #c8f7c6;
        color: black;
    }
    .bg-green-1 {
        background-color: #97e793;
        color: black;
    }
    .bg-green-2 {
        background-color: #66d861;
        color: black;
    }
    .bg-green-3 {
        background-color: #35c72f;
        color: black;
    }
    .bg-green-4 {
        background-color: #04b400;
        color: black;
    }
    .bg-blue {
        background-color: lightblue;
        color: black;
    }
    </style>
    """
    # Create table rows
    rows = ""
    for lst in lists:
        row = "<tr>"
        for item in lst:
            string, integer = item
            if integer == 1:
                row += f"<td class='bg-white'>{string}</td>"
            elif integer == 0:
                row += f"<td class='bg-black'>{string}</td>"
        row += "</tr>"
        rows += row
    # Create the float row
    float_row_html = "<tr>"
    for value in float_row:
        if np.isnan(value):
            float_row_html += "<td class='bg-white'></td>"
        else:
            bucket = min(int(value * 5), 4)
            float_row_html += f"<td class='bg-green-{bucket}'>{value}</td>"
    float_row_html += "</tr>"
    # Create the instance mean agreement row
    num_columns = len(lists[0])
    mean_instance_agr_score_row = f"<tr><td colspan='{num_columns}' class='bg-blue'>{round(mean_instance_agr_score, 2)}</td></tr>"
    table = f"<table>{rows}{float_row_html}{mean_instance_agr_score_row}</table>" # Combine rows into html table
    output = f"{css}{table}" # Combine table and CSS into final html output
    return HTML(output) # Return html object

def save_and_viz(dataset_explanations, index_dataset_instance, range_topks, filename=None):
    """
    :param dataset_explanations: list of explanation objects, one for each instance in the dataset
    :param index_dataset_instance: the index of the instance in the dataset we want to visualize topk machine agr.
    :param range_topks: range of topk values we want to visualize; e.g. top=4 -> [1,2,3,4]
    :param filename: filename in .html format to save output to.
    :return: nothing. Saves html to filename and opens it in a new tab of your webbrowser
    """
    instance_explanations = dataset_explanations[index_dataset_instance]
    html_list = []
    for topk in range(1,range_topks+1):
        mean_instance_agr_score, \
        list_of_masks, \
        agreements_tokenwise, \
        tokens = get_instance_agr_score(instance_explanations, topk=topk)
        list_of_tupled_token_masks = [list(zip(tokens, lm)) for lm in list_of_masks]
        html_list.append(
            colorcode_strings_machine_agreement(lists=list_of_tupled_token_masks,
                                                float_row=[round(agr,2) for agr in agreements_tokenwise],
                                                mean_instance_agr_score=mean_instance_agr_score))
    html_string = "<html><body>"+("<p></p>".join(html.data for html in html_list)) + "</body></html>" #concat html objects
    if filename: #only save to file if filename is provided
        with open(filename, "w") as file: #save to file
            file.write(html_string)
    webbrowser.open_new_tab('file://'+os.path.abspath(filename)) #finds absolute file path and opens in browser

for i in range(10,20):
    save_and_viz(dataset_explanations=test_dataset_explanations,
                 index_dataset_instance=random.randint(0, len(test_dataset_explanations)),
                 range_topks=10,
                 filename=None #filename="./explanations/agreement_machine_viz_"+str(i)+".html"
                 )
"""
Exploration (done on dev set):
- Visualize sentence length bias on top-k agreement
"""
def viz_sent_len_bias():
    """
    function to wrap a lot of code. Visualizes and saves two plots:
        - agreement per group of sentences with same length; 30 subplots from len=11 to len=40
        - agreement per group of sentences with same length; one plot with 7 overlapping functions: len in [11,15,20,25,30,35,40]
    :return: viz and save to file
    """
    sent_lens = [len(x[0].tokens)-3 for x in test_dataset_explanations] # minus [CLS], [SEP], [SEP]
    dataset_unranked_dict_with_sent_len = dict()
    for topk, instance_agr_scores in dataset_unranked_dict.items():
        dataset_unranked_dict_with_sent_len[topk] = list(zip(instance_agr_scores,sent_lens))
    datasets_based_on_len = []
    for sen_len in range(10,41):
        dbol = dict()
        for k,v in dataset_unranked_dict_with_sent_len.items():
            dbol[k] = [agr_score for (agr_score,s_len) in v if s_len == sen_len]
        datasets_based_on_len.append(dbol)
    # Calculate nanmean for each key in the dictionaries
    nanmean_list = []
    for dataset_dict in datasets_based_on_len:
        nanmean_dict = {}
        for entry_key, entry_value in dataset_dict.items():
            nanmean_dict[entry_key] = np.nanmean(entry_value)
        nanmean_list.append(nanmean_dict)
    # Create subplots with adjusted layout
    num_plots = len(datasets_based_on_len)
    num_cols = 5  # Number of columns in the subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Number of rows in the subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 5*num_rows), sharex=True, sharey=True)
    # Iterate over the list of dictionaries and create subplots
    for i, dataset_dict in enumerate(datasets_based_on_len):
        entry_key = list(dataset_dict.keys())[0]
        entry_values = dataset_dict[entry_key]
        nanmean_dict = nanmean_list[i]
        nanmean_vals = [nanmean_dict[key] for key in nanmean_dict.keys()]
        # Calculate the subplot position based on row and column index
        row_idx = i // num_cols
        col_idx = i % num_cols
        # Plot the nanmean values in the specified subplot
        axes[row_idx, col_idx].plot(list(nanmean_dict.keys()), nanmean_vals)
        axes[row_idx, col_idx].set_title("sent_len = "+str(i+10))
        axes[row_idx, col_idx].set_xlabel("top-k")
        axes[row_idx, col_idx].set_ylabel("agreement")
    # Hide empty subplots
    if num_plots < num_rows * num_cols:
        empty_plots = num_rows * num_cols - num_plots
        for i in range(empty_plots):
            row_idx = (num_plots + i) // num_cols
            col_idx = (num_plots + i) % num_cols
            fig.delaxes(axes[row_idx, col_idx])
    # Create an extra plot with overlapping lines for the specified indices
    overlapping_indices = [0, 5, 10, 15, 20, 25, 30]
    # Create the extra plot
    extra_fig, extra_ax = plt.subplots(figsize=(10, 5))
    # Iterate over the specified overlapping indices and plot the lines in the extra plot
    for i, idx in enumerate(overlapping_indices):
        dataset_dict = datasets_based_on_len[idx]
        entry_key = list(dataset_dict.keys())[0]
        entry_values = dataset_dict[entry_key]
        nanmean_dict = nanmean_list[idx]
        nanmean_vals = [nanmean_dict[key] for key in nanmean_dict.keys()]
        # Plot the nanmean values with overlapping lines
        extra_ax.plot(list(nanmean_dict.keys()), nanmean_vals, label="sent_len = " + str(idx + 10))
    extra_ax.set_xlabel("Top-k")
    extra_ax.set_ylabel("Mean agreement@k")
    extra_ax.legend()
    # Save the main plot and the extra plot as separate image files
    main_plot_file = 'explanations/topkagr_per_sentlen_11to41.png'
    extra_plot_file = 'explanations/topkagr_per_sentlen_overlapping_lines.png'
    fig.tight_layout()
    fig.savefig(main_plot_file)
    plt.show()
    print("Should be vizzed and saved to file",main_plot_file)
    extra_fig.tight_layout()
    extra_fig.savefig(extra_plot_file)
    plt.show()
    print("Should be vizzed and saved to file", extra_plot_file)

viz_sent_len_bias()

"""
Pairwise agreement
"""
range_topks = 40 #can be set to a lower number

def get_pairwise_combinations(iterable):
    pairwise_combinations = []
    for el_a in iterable:
        for el_b in iterable:
            if el_a == el_b: # ignore perfect agreement between same method
                continue
            if (el_a, el_b) and (el_b, el_a) not in pairwise_combinations:
                pairwise_combinations.append((el_a,el_b))
    return pairwise_combinations
pairwise_combinations = get_pairwise_combinations(iterable=range(6))

pairwise_dataset_unranked_dicts = []
for pair_idx in pairwise_combinations: #there are 6 attribution methods used in each ferret explanation object
    every_idx_but_the_pair = [idx for idx in range(6) if idx not in pair_idx]
    print("pair_idx:",pair_idx)
    print("every_idx_but_the_pair:",every_idx_but_the_pair," --> these are left out")
    dataset_unranked_dict_pairwise = dict()
    for topk in range(1, range_topks+1):
        dataset_unranked_dict_pairwise[topk] = get_dataset_agr_scores(dataset_explanations=test_dataset_explanations,
                                                                      topk=topk,
                                                                      ranked=False,
                                                                      rm_special_tokens=True,
                                                                      class_1_agreement=True,
                                                                      leaveout_idx_list=every_idx_but_the_pair,
                                                                      dataset_human_aggreg_values=[],
                                                                      dynamic=False)
        print("topk="+str(topk)+" done")
    print("Pairwise: "+str(pair_idx)+" ---done")
    pairwise_dataset_unranked_dicts.append(dataset_unranked_dict_pairwise)

"""
Plotting
 -> pairwise_dataset_unranked_dicts (15)
"""
names = {0:"PartSHAP",1:"LIME",2:"VanGrad",3:"Grad X I",4:"IntGrad",5:"IntGrad X I"}
pairwise_combinations_names = [(names[a],names[b]) for (a,b) in pairwise_combinations]

def plot_pairwise(zoom_topk_range=False):
    if zoom_topk_range:
        datasets = list(
            zip([{k: v for k, v in d.items() if k in zoom_topk_range} for d in pairwise_dataset_unranked_dicts],
                pairwise_combinations_names))
    else:
        datasets = list(zip(pairwise_dataset_unranked_dicts, pairwise_combinations_names))

    fig, ax = plt.subplots(figsize=(10, 8))

    legend_entries = []  # List to store legend entries

    for dataset, title in datasets:
        mean_topks = [np.nanmean(l) for l in dataset.values()]
        mean_topks = [np.nan] + mean_topks  # add mean at 0: nan

        # ['#440154', '#0A5D67', '#FAAA05', '#FFDA80', '#A6A6A6']
        if title == ("PartSHAP", "LIME"):
            marker = "o"
            color = "#FAAA05"
            legend_text = "PartSHAP ~ LIME"
        elif "IntGrad" in title or "Grad X I" in title:
            marker = "s"
            color = "#440154"
            legend_text = "Pairs with IntGrad or Grad X I"
        else:
            marker = "x"
            color = "#0A5D67"
            legend_text = "Other methods"
            # legend_text = "VanGrad | IntGrad X I ~ PartSHAP | LIME\n" \
            #               "VanGrad ~ IntGrad X I"

        # Plot the gray dots at the actual points
        ax.plot(range(len(mean_topks)), mean_topks, marker=marker, color=color, markersize=5, linestyle='--')
        # Plot the gray dot for the mean value at x=1
        ax.plot([1], mean_topks[1], marker=marker, color=color, markersize=5)

        # Store the Line2D object and legend text for each unique entry
        if legend_text not in [entry[1] for entry in legend_entries]:
            legend_entries.append((plt.Line2D([0], [0], marker=marker, color=color, linestyle='--', markersize=5), legend_text))

    ax.plot([], [])  # Dummy plot for the first section
    ax.plot([], [])  # Dummy plot for the second section

    ax.set_xlabel("Top-k")
    ax.set_ylabel("Mean agreement@k")
    # ax.set_title("Method–Method Mean Agreement@k")

    # Create the legend
    legend = ax.legend([entry[0] for entry in legend_entries], [entry[1] for entry in legend_entries], title=None)
    plt.rcParams['font.size'] = 15
    plt.show()

plot_pairwise() #zoomed out
plot_pairwise(zoom_topk_range=range(1,11)) #zoomed in

"""
###################################################
Agreement XAI models - aggregated human annotations
"""
df_test = pd.read_csv("data_original/esnli_test.csv")
df_test = df_test.drop(columns=['Sentence1_Highlighted_1',
                              'Sentence1_Highlighted_2',
                              'Sentence1_Highlighted_3',
                              'Sentence2_Highlighted_1',
                              'Sentence2_Highlighted_2',
                              'Sentence2_Highlighted_3',
                              'Explanation_1',
                              'Explanation_2',
                              'Explanation_3'])

def spanmask(sentence):
    """
    Tokenizer is important here!
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
    # print(tmp_mask)
    # print(mask)
    # print(tokenized_sentence)
    return mask

def create_spanmask_col(sentence_col):
    """
    :param sentence_col: column of dataframe that contains a sentence to compute `spanmask` for
    :return: new column (list) containing the span mask for each sentence
    """
    spanmask_col = []
    for sentence in sentence_col:
        print(sentence)
        if isinstance(sentence,str):
            spanmask_col.append(spanmask(sentence))
        else:
            print(type(sentence), sentence)
            spanmask_col.append(math.nan)
    return spanmask_col

df_test["Sentence12_spanmask_1"] = create_spanmask_col(df_test["Sentence1_marked_1"]+" "+df_test["Sentence2_marked_1"])
df_test["Sentence12_spanmask_2"] = create_spanmask_col(df_test["Sentence1_marked_2"]+" "+df_test["Sentence2_marked_2"])
df_test["Sentence12_spanmask_3"] = create_spanmask_col(df_test["Sentence1_marked_3"]+" "+df_test["Sentence2_marked_3"])

def get_human_aggreg_values(spanmask1,spanmask2,spanmask3):
    """
    :param spanmask1:   [1,0,1,1,1]
    :param spanmask2:   [1,0,0,0,1]
    :param spanmask3:   [1,0,1,0,1]
    :return:            [3,0,2,1,3]
    """
    aggreg_mask = [sum(tup) for tup in zip(spanmask1,spanmask2,spanmask3)]
    return aggreg_mask

df_test["human_aggreg_values"] = [get_human_aggreg_values(l1,l2,l3)
                                  for l1,l2,l3 in zip(df_test["Sentence12_spanmask_1"],
                                                      df_test["Sentence12_spanmask_2"],
                                                      df_test["Sentence12_spanmask_3"])]

test_dataset_human_aggregate_values = df_test["human_aggreg_values"].tolist()

#plot frequence 3-2-1s
def plot_frequence_3s_2s_1s():
    flattened_hum_agg_vals = []
    for vec in test_dataset_human_aggregate_values:
        for score in vec:
            if score != 0:
                flattened_hum_agg_vals.append(score)

    counter = Counter(flattened_hum_agg_vals)
    aggreg_value = list(counter.keys())
    frequencies = list(counter.values())
    plt.bar(aggreg_value, frequencies)
    plt.xlabel('Aggreg value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution Human Aggregated Values')
    plt.xticks([1.0, 2.0, 3.0], [1, 2, 3])
    plt.show()
plot_frequence_3s_2s_1s()

#compute pairwise agreement human-<0,1,2,3,4,5>
human_machine_pairwise_dataset_unranked_dicts = []
for attrib_idx in range(6): #there are 6 attribution methods used in each ferret explanation object
    every_idx_but_the_attrib_idx = [idx for idx in range(6) if idx != attrib_idx]
    print("pair: human -",attrib_idx)
    print("every_idx_but_the_attrib_idx:",every_idx_but_the_attrib_idx," --> these are left out")
    human_machine_dataset_unranked_dict_pairwise = dict()
    for topk in range(1, range_topks+1):
        human_machine_dataset_unranked_dict_pairwise[topk] = get_dataset_agr_scores(
                                                                dataset_explanations=test_dataset_explanations,
                                                                topk=topk,
                                                                ranked=False,
                                                                rm_special_tokens=True,
                                                                class_1_agreement=True,
                                                                leaveout_idx_list=every_idx_but_the_attrib_idx,
                                                                dataset_human_aggreg_values=test_dataset_human_aggregate_values)
        print("topk="+str(topk)+" done")
    print("Pairwise: "+str(attrib_idx)+" ---done")
    human_machine_pairwise_dataset_unranked_dicts.append(human_machine_dataset_unranked_dict_pairwise)

"""
Function needed later
"""
def get_scores_nocls(dataset_explanations, method_i, dataset_human_aggreg_values, strictness_y=-1):
    """
    :param dataset_explanations:
    :param method_i: index from range [0,5]; each index corresponds to a different attribution method
    :param dataset_human_aggreg_values:
    :param strictness_y: if == -1, only returns the list of attrib scores and no gold topk label
    :return: list of tuples (instance scores without CLS/SEP, gold topk from annotations based on strictness)
    """
    dataset_noCLS_scores_method_i = []
    dataset_noCLS_gold_topks = []
    for instance in dataset_explanations:
        instance_noCLS_scores_method_i = [s for s, t in zip(instance[method_i].scores, instance[method_i].tokens)
                                         if t not in {"[CLS]","[SEP]","<s>","</s>","Ġ"}]
        dataset_noCLS_scores_method_i.append(instance_noCLS_scores_method_i)

    if strictness_y == -1:
        return dataset_noCLS_scores_method_i

    else:
        for instance in dataset_human_aggreg_values:
            instance_noCLS_gold_topk = len([s for s in instance if s >= strictness_y])
            dataset_noCLS_gold_topks.append(instance_noCLS_gold_topk)

        return list(zip(dataset_noCLS_scores_method_i, dataset_noCLS_gold_topks))

"""
Plotting
 -> human_machine_pairwise_dataset_unranked_dicts (6)
"""
def plot_pairwise_with_human(zoom_topk_range=False):
    if zoom_topk_range:
        datasets = list(zip([{k: v for k, v in d.items() if k in zoom_topk_range} for d in
                             human_machine_pairwise_dataset_unranked_dicts],
                            ["human ~ " + m for m in ["PartSHAP",
                                                      "LIME",
                                                      "VanGrad",
                                                      "Grad X I",
                                                      "IntGrad",
                                                      "IntGrad X I"]]))

    else:
        datasets = list(zip(human_machine_pairwise_dataset_unranked_dicts,
                            ["human ~ " + m for m in ["p_SHAP",
                                                      "LIME",
                                                      "VanGrad",
                                                      "Grad x I",
                                                      "IntGrad",
                                                      "IntGrad x I"]]))
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = ['o', 's', '^', '*', 'x', 'D']  # Define different markers here
    # colors = ['#440154', '#29788E', '#FAAA05', '#000000', '#008000', '#C4C4C4']
    colors = ['#FAAA05', '#FAAA05', '#29788E',  '#008000', '#008000', '#008000']

    legend_entries = []

    for i, (dataset, title) in enumerate(datasets):
        mean_topks = [np.nanmean(l) for l in dataset.values()]
        mean_topks = [np.nan] + mean_topks  # add mean at 0: nan
        line, = ax.plot(mean_topks, color=colors[i], marker=markers[i], linestyle='--', label=title)
        legend_entries.append((line, title, mean_topks[1]))  # Append tuple with Line2D, title, and value at x=1

    # Sort the legend entries based on the value at x=1
    legend_entries.sort(key=lambda entry: entry[2], reverse=True)

    # Create the legend handles and labels
    handles = [entry[0] for entry in legend_entries]
    labels = [entry[1] for entry in legend_entries]

    plt.rcParams['font.size'] = 15
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Mean agreement@k")
    # ax.set_title("Human—Method Mean Agreement@k")
    ax.legend(handles, labels)
    plt.show()

plot_pairwise_with_human() #zoomed out
plot_pairwise_with_human(zoom_topk_range=range(1,11)) #zoomed in

"""
dynamic top-k experiment with local maxima
"""
def get_local_maxima(list_of_floats):
    """
    Computes the local maxima of a list of floats and returns respective indices.
    Algorithm:
        Point is local maxima if greater than its strict left and right neighbor (except points at index = 0|-1,
        which should only be greater than right or left strict neighbor, respectively) and if greater or equal than a
        threshold. Threshold is mean of the distribution.
    :param list_of_floats:  e.g. list of attribution values
    :return:                array of indices of local maxima
    """
    try:
        mean = np.mean(list_of_floats)
        std = np.std(list_of_floats)
        threshold = mean - (0 * std)  # Set threshold as mean - n * standard deviation

        roll_left = np.roll(list_of_floats, 1)
        roll_right = np.roll(list_of_floats, -1)
        indices = np.where((roll_left < list_of_floats) & (roll_right < list_of_floats) & (list_of_floats >= threshold))[0]

        additional_indices = []

        if list_of_floats[0] > list_of_floats[1] and list_of_floats[0] >= threshold:
            additional_indices.append(0)
        if list_of_floats[-1] > list_of_floats[-2] and list_of_floats[-1] >= threshold:
            additional_indices.append(len(list_of_floats) - 1)

        # Check for spikes with middle point as local maximum
        i = 1
        while i < len(list_of_floats) - 1:
            if list_of_floats[i] >= threshold:
                j = i
                while j < len(list_of_floats) - 1 and list_of_floats[j] == list_of_floats[j + 1]:
                    j += 1
                if j > i:
                    # Check if the cluster is attached to a higher peak without lower points in between
                    try:
                        if (list_of_floats[i - 1] < list_of_floats[i]) and (list_of_floats[j + 1] < list_of_floats[i]):
                            # Find the middle point of the cluster and mark it as local maximum
                            middle_idx = i + (j - i) // 2
                            additional_indices.append(middle_idx)
                        else:
                            # Skip the cluster if it is attached to a higher peak without lower points in between
                            pass
                    except IndexError:
                        print("Skipped error") # --> skips one long sentence which threw an error for some reason
                    i = j
            i += 1

        indices = list(set(indices.tolist() + additional_indices))
        indices.sort()

        return np.array(indices)
    except Exception as e:
        print("Error:", e)
        return np.array([])

def compute_and_plot_local_maxima(list_of_floats, plot=True):
    """
    Compute local maxima of a list of floats
    :param list_of_floats:  e.g. list of attribution values
    :param plot:            defaults to True    --> plots the curves with local maxima
    :return:                tuple of 2          --> (list of floats , local maxima indices)

    #example
    d = [0.1,0.5,0.8,0.2,0.3,0.1,0.5,0.6,0.6,0.1]
    compute_and_plot_local_maxima(d)
    """
    local_maxima_indices = get_local_maxima(list_of_floats)
    if plot:
        plt.plot(list_of_floats)
        plt.plot(local_maxima_indices, [list_of_floats[i] for i in local_maxima_indices], 'ro')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Peaks in Data')
        plt.show()
    return list_of_floats, local_maxima_indices

# overall agreement between methods on dynamic topk (1 value)
dyn_topk_dataset_agr_scores = get_dataset_agr_scores(dataset_explanations=test_dataset_explanations,
                                                     topk=-1,
                                                     ranked=False,
                                                     rm_special_tokens=True,
                                                     class_1_agreement=True,
                                                     leaveout_idx_list=[],
                                                     dataset_human_aggreg_values=[],
                                                     dynamic=True)
dyn_topk_dataset_mean_agr = np.nanmean(dyn_topk_dataset_agr_scores)

# pairwise agreement between methods on dynamic topk (15 values)
pairwise_combinations = get_pairwise_combinations(range(6))
dyn_topk_dataset_mean_agr_pairwise_list = []
for pair_idx in pairwise_combinations: #there are 6 attribution methods used in each ferret explanation object
    every_idx_but_the_pair = [idx for idx in range(6) if idx not in pair_idx]
    print("pair_idx:",pair_idx)
    print("every_idx_but_the_pair:",every_idx_but_the_pair," --> these are left out")
    dyn_topk_dataset_mean_agr_pairwise_list.append(
        np.nanmean(get_dataset_agr_scores(dataset_explanations=test_dataset_explanations,
                                          topk=-1,
                                          ranked=False,
                                          rm_special_tokens=True,
                                          class_1_agreement=True,
                                          leaveout_idx_list=every_idx_but_the_pair,
                                          dataset_human_aggreg_values=[],
                                          dynamic=True),
                   ),
    )
    print("Pairwise: "+str(pair_idx)+" ---done")

# pairwise agreement between methods and human aggreg values on dynamic topk (6 values)
dyn_topk_dataset_mean_agr_pairwise_human_list = []
for attrib_idx in range(6): #there are 6 attribution methods used in each ferret explanation object
    every_idx_but_the_attrib_idx = [idx for idx in range(6) if idx != attrib_idx] #to leave out all but one
    print("pair: human -",attrib_idx)
    print("every_idx_but_the_attrib_idx:",every_idx_but_the_attrib_idx," --> these are left out")
    dyn_topk_dataset_mean_agr_pairwise_human_list.append(
        np.nanmean(get_dataset_agr_scores(dataset_explanations=test_dataset_explanations,
                                       topk=-1,
                                       ranked=False,
                                       rm_special_tokens=True,
                                       class_1_agreement=True,
                                       leaveout_idx_list=every_idx_but_the_attrib_idx,
                                       dataset_human_aggreg_values=test_dataset_human_aggregate_values,
                                       dynamic=True),
                   ),
    )
    print("Pairwise: "+str(attrib_idx)+" ---done")

"""
Figure 1
"""
def figure1(idx_instance, idx_method, hide_axes=True):
    x = test_dataset_explanations[idx_instance]
    attribs = [s for s, t in zip(x[idx_method].scores, x[idx_method].tokens) if t not in ["[CLS]", "[SEP]"]]
    tokens = [t for s, t in zip(x[idx_method].scores, x[idx_method].tokens) if t not in ["[CLS]", "[SEP]"]]

    attribs, loc_maxima_indices = compute_and_plot_local_maxima(list_of_floats=attribs, plot=False)
    print("attribs:",[round(a,2) for a in attribs])

    plt.figure()
    bars = plt.bar(range(len(tokens)), attribs, width=0.9)  # Set width to 1.0 to remove gaps between bars
    plt.xticks(range(len(tokens)), tokens, rotation=55)  # Set x-tick labels to tokens

    mean_attribs = np.mean(attribs)
    plt.axhline(mean_attribs, linestyle='dotted', color='red')
    plt.text(len(tokens) - 1, mean_attribs, 'Mean', ha='right', va='bottom', color='red', fontsize=9)

    # Plotting bars at indices top3, top5, and top7 in different colors
    for i in range(len(tokens)):
        bars[i].set_color("lightblue")

    for idx in loc_maxima_indices:
        bars[idx].set_color('darkgoldenrod')

    indices_attribs = list(enumerate(attribs))
    indices_attribs.sort(key=lambda tup: (tup[1], random.random()))
    top_indices = [i for i, a in indices_attribs]
    top3 = top_indices[-3:]
    top5 = top_indices[-5:]
    top7 = top_indices[-7:]

    d_i = dict()
    for i in range(len(attribs)):
        if i in top3:
            d_i[i] = "3\n5\n7"
        elif i in top5:
            d_i[i] = "5\n7"
        elif i in top7:
            d_i[i] = "7"
    for i, text in d_i.items():
        plt.text(i, attribs[i], text, ha='center', va='bottom', fontsize=9, color='#1f77b4', weight="bold")

    if hide_axes:
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        #plt.xticks([])
        plt.yticks([])
        plt.title('')

    legend_elements = [
        Patch(facecolor='darkgoldenrod', label='Dynamic k'),
        Line2D([], [], linestyle='none', marker='$3$', label="Fixed k=3"),
        Line2D([], [], linestyle='none', marker='$5$', label="Fixed k=5"),
        Line2D([], [], linestyle='none', marker='$7$', label="Fixed k=7")
    ]

    # Adjust the position of the legend
    plt.subplots_adjust(bottom=0.18)  # Add whitespace at the bottom
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.11))

    plt.rcParams['font.size'] = 10
    plt.show()
figure1(3279,2,hide_axes=True)

"""
Other example, same style as figure 1
"""
figure1(1105,4,hide_axes=True)

"""
Figure discussion spans/clusters.
As an illustration for future research, we assign points to clusters through simple distance-based assignment. It 
assigns each point in the series to one of the two clusters based on the distances to the two local maxima 
(local_maxima[0] and local_maxima[1]). The distances are calculated as the sum of the absolute differences between the 
point's value and the centroid value, and the absolute differences between the point's index and the centroid's index. 
The point is assigned to the cluster with the smaller distance. It can be considered as a basic clustering algorithm for 
assigning points to clusters based on distance and index values (e.g. as a simple baseline).
"""
def figure_discussion_spans(idx_instance, idx_method):
    x = test_dataset_explanations[idx_instance]
    attribs = [s for s, t in zip(x[idx_method].scores, x[idx_method].tokens) if t not in ["[CLS]", "[SEP]"]]
    tokens = [t for s, t in zip(x[idx_method].scores, x[idx_method].tokens) if t not in ["[CLS]", "[SEP]"]]

    attribs, loc_maxima_indices = compute_and_plot_local_maxima(list_of_floats=attribs, plot=False)
    attribs_per_cluster = dict()
    attribIndices_per_cluster = dict()
    for i,attrib in enumerate(attribs):
        min_dist = float("inf")
        assigned_cluster = None

        for j,loc_max_idx in enumerate(loc_maxima_indices):
            loc_max = attribs[loc_max_idx]
            dist = abs(attrib - loc_max) + abs(i - loc_max_idx)
            if dist < min_dist:
                min_dist = dist
                assigned_cluster = j
        if assigned_cluster in attribs_per_cluster:
            attribs_per_cluster[assigned_cluster].append(attrib)
            attribIndices_per_cluster[assigned_cluster].append(i)
        else:
            attribs_per_cluster[assigned_cluster] = [attrib]
            attribIndices_per_cluster[assigned_cluster] = [i]


    cluster_colors = [random.choice(['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'brown',
                                     'cyan', 'magenta', 'gray', 'lime', 'teal', 'gold', 'navy'])
                      for _ in attribIndices_per_cluster]
    # cluster_colors = ["blue","gold","green"] #used for figure in paper
    #didn't test if it works when the number of clusters is greater than the number of colors in the list above

        for cluster_idx, cluster_attribs in attribs_per_cluster.items():
            loc_max_idx = loc_maxima_indices[cluster_idx]
            loc_max = attribs[loc_max_idx]
            cluster_color = cluster_colors[cluster_idx]

            plt.plot(loc_max_idx, loc_max, 'o', color=cluster_color, label=f'Cluster {loc_max_idx + 1} Centroid')
            plt.plot(attribIndices_per_cluster[cluster_idx], cluster_attribs, '.', color=cluster_color)

            # Color coding the area under each cluster
            for i in range(len(attribIndices_per_cluster[cluster_idx]) - 1):
                plt.fill_between(attribIndices_per_cluster[cluster_idx][i:i + 2],
                                 cluster_attribs[i:i + 2], alpha=0.3, color=cluster_color)

            # Add labels to the centroids
            plt.text(loc_max_idx, loc_max, f'Span {cluster_idx + 1}', ha='right', va='bottom')


        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        # plt.xticks([])
        plt.xticks(range(len(tokens)), tokens, rotation=55)
        plt.yticks([])
        #plt.xlabel('Index')
        #plt.ylabel('Attribution score')
        #plt.title('Towards span-based explanations\n')
        plt.subplots_adjust(bottom=0.15)  # Add whitespace at the bottom
        plt.show()

        return attribs_per_cluster, attribIndices_per_cluster
figure_discussion_spans(3279,2)

"""
Figure 3
Compare dynamic k to k=4
"""
def make_confusion_matrix_mixed(combinations, scores_l, scores_u, title=None):
    if scores_l:
        # Extract unique classes from the combinations
        classes = sorted(set([item for sublist in combinations for item in sublist]))
        # Generate all possible combinations of classes
        all_combinations = list(product(classes, repeat=2))
        # Create an empty confusion matrix
        confusion_matrix_l = np.zeros((len(classes), len(classes)))
        # Fill the confusion matrix with the scores
        for combination, score in zip(combinations, scores_l):
            actual_class = combination[0]
            predicted_class = combination[1]
            actual_index = classes.index(actual_class)
            predicted_index = classes.index(predicted_class)
            confusion_matrix_l[actual_index, predicted_index] = score
        # Set the missing combinations to the same score as their reverse combinations
        for combination in all_combinations:
            if combination not in combinations and combination[::-1] in combinations:
                reverse_index = combinations.index(combination[::-1])
                score = scores_l[reverse_index]
                actual_class = combination[0]
                predicted_class = combination[1]
                actual_index = classes.index(actual_class)
                predicted_index = classes.index(predicted_class)
                confusion_matrix_l[actual_index, predicted_index] = score
        # Set the special cases where combination is (x, x) with score 1.0
        for i, class_name in enumerate(classes):
            class_index = classes.index(class_name)
            confusion_matrix_l[class_index, class_index] = 1.0

    if scores_u:
        # Extract unique classes from the combinations
        classes = sorted(set([item for sublist in combinations for item in sublist]))
        # Generate all possible combinations of classes
        all_combinations = list(product(classes, repeat=2))
        # Create an empty confusion matrix
        confusion_matrix_u = np.zeros((len(classes), len(classes)))
        # Fill the confusion matrix with the scores
        for combination, score in zip(combinations, scores_u):
            actual_class = combination[0]
            predicted_class = combination[1]
            actual_index = classes.index(actual_class)
            predicted_index = classes.index(predicted_class)
            confusion_matrix_u[actual_index, predicted_index] = score
        # Set the missing combinations to the same score as their reverse combinations
        for combination in all_combinations:
            if combination not in combinations and combination[::-1] in combinations:
                reverse_index = combinations.index(combination[::-1])
                score = scores_u[reverse_index]
                actual_class = combination[0]
                predicted_class = combination[1]
                actual_index = classes.index(actual_class)
                predicted_index = classes.index(predicted_class)
                confusion_matrix_u[actual_index, predicted_index] = score
        # Set the special cases where combination is (x, x) with score 1.0
        for i, class_name in enumerate(classes):
            class_index = classes.index(class_name)
            confusion_matrix_u[class_index, class_index] = 1.0

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a color map for the heatmap
    # cmap_l = colors.LinearSegmentedColormap.from_list('custom',
    #                                                 ['#440154', '#0A5D67', '#FAAA05', '#FFDA80', '#A6A6A6'],
    #                                                 N=1000)
    cmap_l = colors.LinearSegmentedColormap.from_list('custom',
                                                      ['#440154', '#0A5D67', '#FAAA05', '#FFDA80'],
                                                      N=1000)
    cmap_l.set_under('#000000')
    cmap_l.set_over('#000000')
    norm_l = colors.Normalize(vmin=0.45, vmax=1.0, clip=True)
    min_value_l = np.min(confusion_matrix_l)
    max_value_l = np.max(confusion_matrix_l)
    # Create a mask for the upper triangular region
    mask_l = np.tril(np.ones_like(confusion_matrix_l, dtype=bool), k=0)
    mask_u = np.triu(np.ones_like(confusion_matrix_u, dtype=bool), k=0)
    # Plot the confusion matrix as a heatmap with the mask
    sns.heatmap(confusion_matrix_l, annot=True, cmap=cmap_l, fmt=".2f", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax, norm=norm_l, vmin=min_value_l, vmax=max_value_l,
                mask=mask_u)

    cmap_u = colors.ListedColormap(['black', '#A6A6A6'])
    cmap_u.set_under('#ffffff')
    cmap_u.set_over('#ffffff')
    norm_u = colors.Normalize(vmin=-0.099999999, vmax=0.0999999999, clip=True)
    # Plot the upper triangular region with the new colormap
    sns.heatmap(confusion_matrix_u, annot=True, cmap=cmap_u, fmt=".2f", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax, norm=norm_u,
                vmin=-0.099999999,
                vmax=0.0999999999,
                mask=mask_l,
                cbar_kws={'ticks': [0.0]})

    # Set the axis labels and title
    # ax.set_xlabel('Attrib. Method or Human')
    # ax.set_ylabel('Attrib. Method or Human')
    if title:
        ax.set_title(title, fontstyle='italic')

    # Rotate y-axis tick labels
    plt.setp(ax.get_yticklabels(), rotation=30, ha='right')
    # Rotate x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    # Show the plot
    plt.rcParams['font.size'] = 10
    plt.show()

names = {0:"PartSHAP",1:"LIME",2:"VanGrad",3:"Grad X I",4:"IntGrad",5:"IntGrad X I"}
all_dyn_topk_pairwise_combinations = [(names[i],names[j]) for (i,j) in pairwise_combinations] + \
                                     [("human",name) for name in names.values()]
all_dyn_topk_pairwise_results = dyn_topk_dataset_mean_agr_pairwise_list + dyn_topk_dataset_mean_agr_pairwise_human_list

pairwise_means_at_4 = [np.nanmean(pairwise_d[4]) for pairwise_d in pairwise_dataset_unranked_dicts]
pairwise_means_at_4_human = [np.nanmean(pairwise_d[4]) for pairwise_d in human_machine_pairwise_dataset_unranked_dicts]
pairwise_plusmin_differences_dyn_4 = []
for dynk,fixk in zip(all_dyn_topk_pairwise_results, pairwise_means_at_4+pairwise_means_at_4_human):
    improvement = dynk-fixk
    pairwise_plusmin_differences_dyn_4.append(improvement)

make_confusion_matrix_mixed(combinations=all_dyn_topk_pairwise_combinations,
                            scores_l=all_dyn_topk_pairwise_results,
                            scores_u=pairwise_plusmin_differences_dyn_4)

"""
Further analyses
Compare dynamic k to k=1,2,3,...,10
"""
for FIXED_K in range(1,11):
    pairwise_means_at_fixed_k = [np.nanmean(pairwise_d[FIXED_K]) for pairwise_d in pairwise_dataset_unranked_dicts]
    pairwise_means_at_fixed_k_human = [np.nanmean(pairwise_d[FIXED_K]) for pairwise_d in
                                 human_machine_pairwise_dataset_unranked_dicts]
    pairwise_differences_dyn_fixed_k = []
    for dynk, fixk in zip(all_dyn_topk_pairwise_results, pairwise_means_at_fixed_k + pairwise_means_at_fixed_k_human):
        improvement = max(dynk - fixk, 0)
        pairwise_differences_dyn_fixed_k.append(improvement)

    make_confusion_matrix_mixed(combinations=all_dyn_topk_pairwise_combinations,
                                scores_l=all_dyn_topk_pairwise_results,
                                scores_u=pairwise_differences_dyn_fixed_k,
                                title="dynamic k vs fixed k="+str(FIXED_K))

"""
Counting mean topk per method
"""
ddddd = {0:[],1:[],2:[],3:[],4:[],5:[]}
for x in test_dataset_explanations:
    for i in range(6):
        k_indices = get_local_maxima([s for s,t in zip(x[i].scores,x[i].tokens) if t not in ["[CLS]","[SEP]"]])
        ddddd[i].append(len(k_indices))

for k,v in ddddd.items():
    print(k,np.mean(v))
#0 4.536135993485342
#1 5.340492671009772
#2 4.583469055374593
#3 6.830008143322476
#4 7.299572475570033
#5 5.675386807817589

for k,v in ddddd.items():
    print(k,np.std(v))
#0 1.7329665367041542
#1 2.352662657794251
#2 1.67703748941373
#3 2.5870455587524392
#4 2.6297625906193334
#5 2.371489530796918
