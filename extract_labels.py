#!/usr/bin/python 
# -*- coding: utf-8 -*-

import json
import os
from stanfordcorenlp import StanfordCoreNLP

"""
Function
    Based on aspects: '主题', '剧情', '配乐', '画面', '节奏', '手法', '演技', '整体'
    Extract labels and review indexes for each aspect
IMPORTANT
    entity.json
    reviews_chinese.json
    stanford-corenlp-full-2018-10-05
    should be placed with this py file
    
    stanford-chinese-corenlp-2018-10-05-models.jar
    should be placed in stanford-corenlp-full-2018-10-05
Input
    entity: dict
        entity['aspect']['剧情', ...] = [enlarged words similar to 剧情, ... after my deletion]
        entity['adj']['正面形容'/'负面形容']
        entity['verb']['正面动词'/'负面动词']
        entity['adv']['副词']
    reviews: list of list
        some reviews with only divided Chinese words
        reviews = [sentence1, sentence2, ...]
        sentence = [Chinese_word1, Chinese_word2, ...]
Output
    labels: dict of dict
        dictionary containing potantial tags extracted from reviews
        appear one time, add one time, has overlap
        labels['aspect']['剧情', ...] = [(potential_tag_for_剧情, corresponding_review_index), ...]
        tag may not in entity
        labels['sentiment']['正面'/'负面'] = [(potential_tag_for_电影, corresponding_review_index), ...]
        tag must in entity['adj']['正面形容'/'负面形容']
        Printed and they can be saved as json
"""

current_path = os.path.dirname(__file__)
entity = json.load(open(current_path+'\\entity.json', 'rb'))
reviews = json.load(open(current_path + "\\reviews_chinese.json", "rb"))


def extract_labels(threshold:int=90000, save:bool=True):
    nlp = StanfordCoreNLP(current_path + '\\stanford-corenlp-full-2018-10-05', lang='zh', memory='8g')
    aspect = {'主题':[], '剧情':[], '配乐':[], '画面':[], '节奏':[], '手法':[], '演技':[]}
    sentiment = {'正面':[], '负面':[]}
    arg_aspect = {}
    for key, values in entity['aspect'].items():
        for value in values:
            arg_aspect[value] = key

    # extract labels
    for i in range(threshold):
        # redivide words and analyze part of speeches
        review = ''
        for c in reviews[i]: review += c
        if review == '': continue
        words = nlp.word_tokenize(review)
        part_of_speech = nlp.pos_tag(review)
        
        try:
            # 1. labels['aspect'] = comments(adj) on 剧情, ...
            for speech in part_of_speech:
                if speech[1] == 'VA':
                    if speech[0] in entity['adj']['正面形容']:
                        sentiment['正面'].append((speech[0], i))
                    if speech[0] in entity['adj']['负面形容']:
                        sentiment['负面'].append((speech[0], i))
            
            # 2. labels['sentiment']['剧情', ...] = sentimental adj in entity['adj']
            for dp in nlp.dependency_parse(review):
                if dp[0] == 'nsubj' and words[dp[2]-1] in arg_aspect and part_of_speech[dp[1]-1][1] == 'VA':
                    aspect[arg_aspect[words[dp[2]-1]]].append((words[dp[1]-1], i))
            
            # 3. output dynamically
            if i % 10 == 0:
                print("Extracting labels:", round(i/threshold*100, 2), "%")
        except json.decoder.JSONDecodeError: continue
    nlp.close()
    
    # save labels
    labels = {'aspect':aspect, 'sentiment':sentiment}
    if save == True:
        print("Saving labels_" + str(threshold) + '.json ...')
        with open(current_path + '\\labels_' + str(threshold) + '.json', 'w') as f:
            json.dump(labels, f)

if __name__ == "__main__":
    extract_labels(200000)