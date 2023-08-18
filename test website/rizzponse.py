# get pick up lines from data.txt and generate a new pick up line


import random
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download()
from pywsd.utils import lemmatize_sentence
import tensorflow_hub as hub


# read in data
with open('data.txt', 'r') as f:
    data = f.read()

    #split data into lines 
    lines = data.split('\n')

    lines = [line.replace('"', '') for line in lines]

    lines = [line for line in lines if len(line) > 0]

    lines1 = lines

    #remove trailing commas
    lines = [line[:-1] if line[-1] == ',' else line for line in lines]

    #make all lines lowercase
    lines = [line.lower() for line in lines]

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    #remove stopwords from lines
    for (i, line) in enumerate(lines):
        words = word_tokenize(line)
        words = [word for word in words if word not in stop_words]
        # remove punctuations from the line
        words = [word for word in words if word.isalpha()]
        lines[i] = ' '.join(words)
    
    # lemmatize all sentences in lines
    lemmatized_lines = [lemmatize_sentence(line) for line in lines]

    # use jaccaard similarity to find similarity between user input and lines
    # note that lemmaized_user_input is a list of words and lemmatized_lines is a list of lists of words

    # jaccaard similarity function
    def jaccard_similarity(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union
    
    # # user input 
    # user_input = input("Enter a sentence: ")
    # user_input = user_input.lower()

    # # lemmatize user input
    # lemmatized_user_input = lemmatize_sentence(user_input)
    
    # # find similarity between user input and lines
    # similarities = [jaccard_similarity(lemmatized_user_input, line) for line in lemmatized_lines]

    # # find the index of the most similar line
    # most_similar_index = np.argmax(similarities)
    # print (most_similar_index)

    # # print the most similar line in the format as stored in the data.txt file
    # print (lines1[most_similar_index])

    # jaccaard similarity function
    def generate_pickup_line(user_input):
        user_input = user_input.lower()
        lemmatized_user_input = lemmatize_sentence(user_input)
        similarities = [jaccard_similarity(lemmatized_user_input, line) for line in lemmatized_lines]
        most_similar_index = np.argmax(similarities)
        # if there is no similarity between user input and lines, return none
        if similarities[most_similar_index] == 0:
            return None
        return lines1[most_similar_index]
    
    # use tdf-idf to find similarity between user input and lines
    # note that lines is a list of strings and user_input is a string


    # tdf-idf similarity function
    def generate_pickup_line_tfidf(user_input):
        user_input = user_input.lower()
        lines = lines1
        lines.append(user_input)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(lines)
        similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])
        most_similar_index = np.argmax(similarity_matrix)
        # if there is no similarity between user input and lines, return none
        if similarity_matrix[0][most_similar_index] == 0:
            return None
        return lines1[most_similar_index]
    
    # Universal Sentence Encoder (USE) similarity function - Google
    # note that lines is a list of strings and user_input is a string
    def generate_pickup_line_USE(user_input):
        user_input = user_input.lower()
        lines = lines1
        lines.append(user_input)
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        vectors = embed(lines)
        similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])
        most_similar_index = np.argmax(similarity_matrix)
        # if there is no similarity between user input and lines, return none
        if similarity_matrix[0][most_similar_index] == 0:
            return None
        return lines1[most_similar_index]
    
    
    #     Word2vec is great at understanding the individual word, vectorizing the whole sentence takes a long time. Let alone the entire document.

    # Instead we will use Doc2vec — a similar embedding algorithm that vectorizes paragraphs instead of each word (2014, Google Inc). In a more digestible format, you can check out this intro blog by Gidi Shperber.

    # Unfortunately for Doc2vec, no corporation sponsored pretrained model has been published. We will use pretrained enwiki_dbow model from this repo. It is trained on English Wikipedia (unspecified number but the model size is decent at 1.5gb).


    
    # input in a loop, exit loop when user clicks esc
    while True:
        user_input = input("Enter a sentence: ")
        if user_input == 'esc':
            break
        else:
            print (generate_pickup_line(user_input))
            print (generate_pickup_line_tfidf(user_input))
            print (generate_pickup_line_USE(user_input))






    






    









    

    



    



   


    

























