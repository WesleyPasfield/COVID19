import numpy as np
import pandas as pd
import re
import json
import sys
import os
import ast
import random
import nltk
import gensim
import wordcloud
import faiss
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import gensim
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

class process_articles:
    
    def __init__(self, file_locs, stopwords):
    
        self.file_locs = file_locs
        self.stopwords = stopwords
    
    def read_files(self):
        
        self.title_text = []
        
        for file_loc in self.file_locs:
        
            ## Get all files from specified location
            print('Processing Files at the below location:')
            print(file_loc)
            self.raw_root_files = os.listdir(file_loc)
            
            # Randomly sample files to reduce training file size - 10% of files from each supply
            self.root_files = []
            for val in np.arange(int(len(self.raw_root_files) / 5)):
                
                self.root_files.append(random.choice(self.raw_root_files))
            
            print('There are {} files to process'.format(len(self.root_files)))
            print('There were {} files in the dataset'.format(len(self.raw_root_files)))
            file_root = file_loc.split('/')[1]

            ## Loop through each file and grab the title and text
    
            for file in self.root_files:

                with open('{}/{}'.format(file_loc, file)) as f:
                    art_text_fin = []

                    try:
                        ## Load article and extract title
                        article = json.load(f)
                        art_title = article['metadata']['title']

                        ## Text is stored in multiple blocks - loop through each one
                        art = article['body_text']
                        art_text = []
                        for text in np.arange(len(art)):
                            raw_text = art[text]['text']
                            art_text.append(raw_text)

                        ## Condense each block together in a single form
                        ## Store raw text and titles in list
                        art_text_fin.append(" ".join(str(text_block) for text_block in art_text))
                        self.title_text.append([file_root, art_title, art_text_fin])

                    except:

                        print('FAILURE !!! \n')
                        print(article)
            
    def process_text(self):
        
        p_stemmer = PorterStemmer()
        articles = [article[2] for article in self.title_text]
        
        ## Process Each document - remove junk
        
        print('Cleaning out Junk')
        
        articles = [str(article).lower() for article in articles]
        articles = [re.sub('<[^<]+?>', '', article) for article in articles]
        articles = [re.sub(r'http\S+', '', article) for article in articles]
        articles = [re.sub(r'[^A-Za-z0-9]+', ' ', article) for article in articles]
        articles = [re.sub(r'\\', '', article) for article in articles]
        articles = [re.sub(r'\[.*?\]', '', article) for article in articles]
        articles = [re.sub(r'\d+', '', article) for article in articles]
        
        ## Tokenize
        ## deacc=True drops out punctuation
        
        print('Tokenizing words')
        
        articles = [gensim.utils.simple_preprocess(str(article), deacc=True) for article in articles]
        articles = [ast.literal_eval(str(article)) for article in articles]
        
        ## Convert into words
        ## Clean out stop words
        
        print('Converting to list of words and removing stop words')
        
        articles = [[word.strip() for word in article] for article in articles] 
        articles = [[word for word in article if word not in self.stopwords] for article in articles]
        
        ## Stem words
        
        print('Creating word stems')
        
        articles = [[p_stemmer.stem(word) for word in article] for article in articles]
        
        self.processed_article = [[tt[0], tt[1], tt[2], article] for tt, article in zip(self.title_text, articles)]

    def bigrams(self, tokenized_articles, min_count=3, threshold=30):
        
        ## Create bigrams from raw tokenized text provided
        
        bigram = gensim.models.Phrases(tokenized_articles, min_count = min_count, threshold = threshold)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        bigram_fin = [bigram_mod[article] for article in tokenized_articles]
        
        self.bigram = bigram_fin
        
    def trigrams(self, tokenized_articles, min_count=3, threshold=30):
        
        print('Creating Bigrams')
        
        ## Create bigrams from raw tokenized text provided
        
        bigram = gensim.models.Phrases(tokenized_articles, min_count = min_count, threshold = threshold)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        
        print('Creating Trigrams from Bigrams')
        
        ## Create trigrams from the bigram model and raw text
        
        trigram = gensim.models.Phrases(bigram[tokenized_articles])
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        trigram_fin = [trigram_mod[bigram_mod[article]] for article in tokenized_articles]
        trigram_fin = [str(trigram) for trigram in trigram_fin]
        
        self.processed_trigrams = [[pa[0], pa[1], pa[2], pa[3], trigram] for pa, trigram in zip(self.processed_article, trigram_fin)]
                    
def train_test_splitter(processed_text, train_prop):
    
    train = processed_text[0:int(len(processed_text) * train_prop)]
    test = processed_text[int(len(processed_text) * 0.8):]
    
    return train, test

class LDA_Evaluator:
    
    def __init__(self, lda_model, vectorizer):
        
        self.lda_model = lda_model
        self.feature_names = vectorizer.get_feature_names()
        self.vectorizer = vectorizer
        
        ## Create DF
        
        components = pd.DataFrame(self.lda_model.components_).copy()
        components['fullsum'] = components.sum(axis=1)
        
        self.components = components
        
        ## Determine Component Contribution 
        
        allwords = self.components['fullsum'].sum()
        self.topic_distro = self.components['fullsum'] / allwords
        
        for col in self.components.columns:
            self.components[col] = self.components[col] / self.components['fullsum']
        self.components.drop(['fullsum'], inplace = True, axis = 1)
        self.components = self.components.transpose()
        self.components['wordmean'] = self.components.mean(axis=1)
        self.components.index = self.feature_names
        
        ## Determine Word Distribution
        
        words = pd.DataFrame(self.lda_model.components_).copy()
        words = words.transpose()
        words['fullsum'] = words.sum(axis=1)
        wordstotal = words['fullsum'].sum()
        word_distribution = words['fullsum'] / wordstotal
        word_rank = word_distribution.rank() / len(word_distribution)
        
        ## Add back to DF
        
        self.components['word_rank'] = word_rank.values
        self.components['word_distro'] = word_distribution.values
        
    def eval_raw_frequency(self, topic, num_words, threshold=0):
        
        ## Returns words that show up most per topic
        
        raw_vals = self.components.copy()
        raw_vals = raw_vals[raw_vals['word_rank'] >= threshold]
        
        return raw_vals.sort_values(by=topic, ascending=False).head(n=num_words)
    
    def eval_rel_frequency(self, topic, num_words, threshold = 0):
        
        ## Returns words that show up disproportionately by topic
        
        rel_freq = self.components.copy()
        rel_freq = rel_freq[rel_freq['word_rank'] >= threshold]
        for col in rel_freq[0:(len(rel_freq.columns)-1)]:
            rel_freq[col] = rel_freq[col] / rel_freq['wordmean'] ## Calc how much higher/lower prop is
        
        return rel_freq.sort_values(by=topic, ascending = False).head(n=num_words)       
        
class wcEval:
    
    def __init__(self, data, vectorizer, ldamod):

        self.term_freq = vectorizer.transform([t[4] for t in data])
        self.topic_scores = pd.DataFrame(ldamod.transform(self.term_freq))
        self.topic_cols = self.topic_scores.columns.values
    
    def raw_freq_wc(self):
        
        test_topics = self.topic_scores.copy()

        test_topics['max'] = test_topics.max(axis=1)
        test_topics['max_topic_num'] = 99
        for col in self.topic_cols:
            test_topics['max_topic_num'] = np.where(test_topics['max'] == test_topics[col], col, test_topics['max_topic_num'])

        primary_topic = test_topics['max_topic_num'].tolist()

        self.raw_primary_topic = primary_topic

    def rel_freq_wc(self):
        
        test_topics = self.topic_scores.copy()

        ## Convert to relative freq

        for col in self.topic_cols:
            test_topics[col] = test_topics[col] / np.mean(test_topics[col])

        ## Find max per column

        test_topics['max'] = test_topics.max(axis=1)
        test_topics['max_topic_num'] = 99

        for col in self.topic_cols:
            test_topics['max_topic_num'] = np.where(test_topics['max'] == test_topics[col], col, test_topics['max_topic_num'])

        primary_topic = test_topics['max_topic_num'].tolist()

        self.rel_primary_topic = primary_topic

def word_clouds(data, topic, max_words, stop_words, evalinfo):
        
    word_cloud = [str(t[1]) for t in data if t[2] == topic]
    print(f'Percentage in topic {topic}: {np.around(len(word_cloud) / len(data), 4)}')
    word_cloud = ' '.join(w for w in word_cloud)
        
    wordcloud = WordCloud(max_words = max_words, stopwords = stop_words).generate(word_cloud)
        
    #plt.set_size_inches(18.5, 10.5)
    plt.figure( figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    print(f'10 highest relative frequency words in topic {topic}:\n')
    print(evalinfo.eval_rel_frequency(topic, 10).index.values)
    print('\n')