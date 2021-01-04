import random
import nltk
# nltk.download()


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest

import math
import networkx
import numpy

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# Sample data
# I have taken the abstract of my final MSc Thesis
# Consists of 10 lines
text = """Federated edge cloud (FEC) is an edge cloud environment where multiple edge servers in a single administrative domain collaborate together to provide real-time services.
As the number of edge servers increases in FEC, the amount of energy consumed by servers and network switches also increases.
This creates another challenge for how to schedule delay-sensitive services over FEC, while minimizing the total energy consumption and reducing the QoS violation of a service at the same time.
This paper proposes an energy-efficient service scheduling mechanism in FEC called ESFEC, which consists of a placement algorithm and three types of reconfiguration algorithms.
Unlike traditional approaches, the placement algorithm in ESFEC places delay-sensitive services on the edge servers in nearby edge domain instead of clouds. In addition, ESFEC schedules services with actual traffic requirements rather than using maximum traffic requirements to ensure QoS. This increases the number of services co-located in a single server and thereby reduces the total energy consumed by the services.
Although this approach is likely to increase the number of service migrations in heavy traffic conditions, ESFEC reduces the migration overhead using a reinforcement learning (Q-learning) based reconfiguration algorithm, ESFEC-RL (reinforcement learning), that can dynamically adapt to a changing environment.
ESFEC also includes two different heuristic algorithms such as ESFEC-EF (energy first) and ESFEC-MF (migration first), which are more suitable for real scale scenarios.
The simulation results show that the ESFEC improves energy efficiency by up to 28% and lowers the service violation rate by up to 66% against a traditional approach used in the edge cloud environment."""


def tokenize_content(content):

    stop_words = set(stopwords.words('english') + list(punctuation))
    # separating the input text by earch word.
    words = word_tokenize(content.lower())
    return (sent_tokenize(content), [word for word in words if word not in stop_words])


def score_tokens(sent_tokens, word_tokens):
    # FreqDist is basically counting words in the "text"
    word_freq = FreqDist(word_tokens)
    # print(word_freq.keys())
    # initializing a dictionary
    rank = defaultdict(int)
    for i, sentence in enumerate(sent_tokens):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                # ranking each sentence in the input text
                rank[i] += word_freq[word]
    # print(rank)
    return rank


def summarize(ranks, sentences):
    # I set 4 as the parameter because I wanted my summary to be 5 sentences long.
    # It can be chnaged
    indices = nlargest(4, ranks, key=ranks.get)
    final_summary = [sentences[j] for j in indices]
    return ' '.join(final_summary)


# print(text)  # contains 10 lines
print("---------------------------------------------------------------------------------------------------------------------------------------------")
# sentence tokens and word tokens, separated by each sentence and each word
sent_tokens, word_tokens = tokenize_content(text)
sent_ranks = score_tokens(sent_tokens, word_tokens)
print(summarize(sent_ranks, sent_tokens))


########################################################
#                                                      #
# This algorithm below also works but                  #
# since this uses a python library called networkx     #
# I am commenting it out                               #
#                                                      #
########################################################

# def getrank(document):

#     # separating each line whenever a "." is detected
#     sentences = PunktSentenceTokenizer().tokenize(document)
#     vectorizer = CountVectorizer()
#     bow_matrix = vectorizer.fit_transform(sentences)
#     # print(vectorizer.get_feature_names())
#     # print(bow_matrix.toarray())
#     # print("---------")

#     normalized = TfidfTransformer().fit_transform(bow_matrix)
#     # print(normalized.toarray())

#     similarity_graph = normalized * normalized.T

#     nx_graph = networkx.from_scipy_sparse_matrix(similarity_graph)
#     values = networkx.pagerank(nx_graph)
#     sentence_array = sorted(
#         ((values[i], s) for i, s in enumerate(sentences)), reverse=True)

#     sentence_array = numpy.asarray(sentence_array)

#     freq_max = float(sentence_array[0][0])
#     freq_min = float(sentence_array[len(sentence_array) - 1][0])

#     temp_array = []
#     for i in range(0, len(sentence_array)):
#         if freq_max - freq_min == 0:
#             temp_array.append(0)
#         else:
#             temp_array.append(
#                 (float(sentence_array[i][0]) - freq_min) / (freq_max - freq_min))

#     threshold = (sum(temp_array) / len(temp_array)) + 0.25

#     sentence_list = []

#     for i in range(0, len(temp_array)):
#         if temp_array[i] > threshold:
#             sentence_list.append(sentence_array[i][1])

#     seq_list = []
#     for sentence in sentences:
#         if sentence in sentence_list:
#             seq_list.append(sentence)

#     return seq_list

# textSummarized = getrank(text)
# print(textSummarized)

######################################## ends ##########################################
