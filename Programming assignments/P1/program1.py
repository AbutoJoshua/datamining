#Joshua Abuto
#1001530342

import math
import os, nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# variables
docdict = dict()  # dictionary of {doc : {word1 : word1_num, word2 : word2_num, .... }}
dicttf = dict()  # Dictionary of {word : [doc1, doc3, ... ]}
wtfidfdict = dict()  # dictionary for weighted tf-idf vector
wqdict = dict()  # dictionary for weight query
tfdictionary = dict()  # tf dictionary
postinglist = dict()  # posting list
num_of_docs = 0
simcos = dict()  # similarity cosine dictionary
stemmed_words = []


def gettokens():
    j = 0
        # reading the code and stemming each token
        for filename in os.listdir(corpusroot):
            file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
            doc = file.read()
            file.close()
            doc = doc.lower()

            # Tokenizing the document
            tokens = tokenizer.tokenize(doc)
            token_list = [s.strip(' ') for s in tokens]
            # print(tokens)
            # print(sorted(tokens))

            # Removing the stop words and the spaces
            filtered_sentence = [w for w in token_list if not w in stop_words]
            filtered_sentence = [w for w in filtered_sentence if not w in '']

            # Stemming the filtered words
            stemmer = PorterStemmer()
            for token in filtered_sentence:
                stemmed_words.append(stemmer.stem(token))
            # print(stemmed_words)

def getidf(tokens):
    # Calculating the TF for each term in each document {doc : {word1 : word1_num, word2 : word2_num, .... }}
    for word in stemmed_words:
        if word in dicttf:
            dicttf[j][word] += 1
        else:
            dicttf[j][word] = 1

    # Calculating the tf for each query for all documents
    for word in stemmed_words:
        if word in tfdictionary:
            tfdictionary[word] += 1
        else:
            tfdictionary[word] = 1

        # {word : [doc1, doc3, ... ]}
        text_str = str(i) + '.txt'
        if word in docdict:
            if text_str not in docdict[word]:
                docdict[word].append(text_str)

        else:
            docdict[word] = list()
            docdict[word].append(text_str)

        j += 1

def weight(filename,token):
    for j in range(0, 30):
        for word in dicttf:
            if word not in dicttf[j][word]:
                wtfidfdict[j][word] = (1 + math.log(dicttf[j][word], 10)) * (math.log(30 / tfdictionary[word]))

        for word in tfdictionary:
            if word not in wqdict:
                wqdict[word] = (1 + math.log(tfdictionary[word]), 10)

        for k in range(0, len(wtfidfdict[j]) - 1):
            sorted(wtfidfdict[j][word], key=word, reverse=True);
    

# I was meant to break down all the methods but I did not want to ruin my code

def main():
    corpusroot = './presidential_debates/presidential_debates'
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'[A-Z|a-z|0-9]*')

    for i in range(0, 30):
        dicttf[i] = dict()
        wtfidfdict[i] = dict()
        simcos[i] = dict()

    j = 0
    # reading the code and stemming each token
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        doc = doc.lower()

        # Tokenizing the document
        tokens = tokenizer.tokenize(doc)
        token_list = [s.strip(' ') for s in tokens]
        # print(tokens)
        # print(sorted(tokens))

        # Removing the stop words and the spaces
        filtered_sentence = [w for w in token_list if not w in stop_words]
        filtered_sentence = [w for w in filtered_sentence if not w in '']

        # Stemming the filtered words
        stemmer = PorterStemmer()
        stemmed_words = []
        for token in filtered_sentence:
            stemmed_words.append(stemmer.stem(token))
        # print(stemmed_words)

        # Calculating the TF for each term in each document {doc : {word1 : word1_num, word2 : word2_num, .... }}
        for word in stemmed_words:
            if word in dicttf:
                dicttf[j][word] += 1
            else:
                dicttf[j][word] = 1

        # Calculating the tf for each query for all documents
        for word in stemmed_words:
            if word in tfdictionary:
                tfdictionary[word] += 1
            else:
                tfdictionary[word] = 1

            # {word : [doc1, doc3, ... ]}
            text_str = str(i) + '.txt'
            if word in docdict:
                if text_str not in docdict[word]:
                    docdict[word].append(text_str)

            else:
                docdict[word] = list()
                docdict[word].append(text_str)

            j += 1
    # going through all documents for calculations
    for j in range(0, 30):
        for word in dicttf:
            if word not in dicttf[j][word]:
                wtfidfdict[j][word] = (1 + math.log(dicttf[j][word], 10)) * (math.log(30 / tfdictionary[word]))

        for word in tfdictionary:
            if word not in wqdict:
                wqdict[word] = (1 + math.log(tfdictionary[word]), 10)

        for k in range(0, len(wtfidfdict[j]) - 1):
            sorted(wtfidfdict[j][word], key=word, reverse=True);

        for l in range(0, 9):
            if word not in postinglist:
                postinglist[word] = word
                postinglist[word][l] = j

        for word in wqdict:
            if word in postinglist:
                if j in postinglist[word]:
                    simcos[j][word] = wqdict[word] * wtfidfdict[j][word]
                else:
                    simcos[j][word] += wqdict[word] * wtfidfdict[j][postinglist[word][9]]

        j += 1

        print(simcos);
