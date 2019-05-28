#!/usr/bin/python 
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from stanfordcorenlp import StanfordCoreNLP

from helpers import *
from calculation import *

"""
Function
    Based on aspects: '主题', '剧情', '配乐', '画面', '节奏', '手法', '演技', '整体'
    Rate each aspect
    Get tags on each aspect
    Recommand reviews on each aspect (positive, negative)
IMPORTANT
    If labels_90000.json is not prepared, extract_labels.py must be run first
    
    helpers.py (basic functions)
    calculation.py (calculation functions)
    should be placed with this py file
    
    entity.json
    embedding_matrix.pickle
    reviews_chinese.json
    reviews_full.json
    labels_90000.json
    stanford-corenlp-full-2018-10-05
    should be placed with this py file
    
    stanford-chinese-corenlp-2018-10-05-models.jar
    should be placed in stanford-corenlp-full-2018-10-05
"""

current_path = os.path.dirname(__file__)
reviews_full = json.load(open(current_path + "\\reviews_full.json", "rb"))


def generate_summary(num_of_sentences:int=10, inclination:str='objective'):
    """
    Function
        Extract the sentence containing relative word of aspect from the review
    Input
        num_of_sentences: number of sentences in each aspect's summary
        inclination: summary inclination
    Output
        summary: dict, specific sentence of each aspect
    """
    labels = json.load(open(current_path+'\\labels_90000.json', 'rb'))
    summary = {}
    rf = trainRF()
    
    if inclination == 'objective':
        print("\n\n生成客观评论摘要\n")
    elif inclination == 'positive':
        print("\n\n生成正面评论摘要\n")
    elif inclination == 'negative':
        print("\n\n生成负面评论摘要\n")
    else: raise RuntimeError('InvalidInclinationError')
    
    for aspect in labels['aspect']:
        print('\n收集' + aspect + '评论...')
        print('评论情感分类')
        # divide tags given sentiment
        labels['aspect'][aspect] = divide(labels['aspect'][aspect], rf)
        if labels['aspect'][aspect][0] == []: labels['aspect'][aspect][0] = [('好', 0)]
        if labels['aspect'][aspect][1] == []: labels['aspect'][aspect][1] = [('差', 0)]
        
        pos_reviews = []
        neg_reviews = []
        # extract corresponding reviews
        for tup in labels['aspect'][aspect][0]:
            pos_reviews.append(reviews_full[tup[1]])
        for tup in labels['aspect'][aspect][1]:
            neg_reviews.append(reviews_full[tup[1]])

        
        # sample reviews given inclination
        paragraph = ''
        if inclination == 'objective':
            # inclination influences summary by score
            score = float(len(labels['aspect'][aspect][0]))/(len(labels['aspect'][aspect][0]) + len(labels['aspect'][aspect][1]))
            print('采集正面评论')
            pos_samples = sample(pos_reviews, int(num_of_sentences*score))
            print('采集负面评论')
            neg_samples = sample(neg_reviews, num_of_sentences - int(num_of_sentences*score))
            # extract the sentence containing relative word of aspect from the review
            for review in pos_samples:
                paragraph += extract_specific_sentence(review, aspect)
            for review in neg_samples:
                paragraph += extract_specific_sentence(review, aspect)
        
        elif inclination == 'positive':
            print('采集正面评论')
            pos_samples = sample(pos_reviews, num_of_sentences)
            for review in pos_samples:
                paragraph += extract_specific_sentence(review, aspect)
        
        else:
            print('采集负面评论')
            neg_samples = sample(neg_reviews, num_of_sentences)
            for review in neg_samples:
                paragraph += extract_specific_sentence(review, aspect)
        summary[aspect] = paragraph
    return summary


if __name__ == "__main__":
    visualization(generate_summary(10, 'objective'))
    visualization(generate_summary(10, 'positive'))
    visualization(generate_summary(10, 'negative'))
    nlp.close()