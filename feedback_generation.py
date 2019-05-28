#!/usr/bin/python 
# -*- coding: utf-8 -*-

import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

standard = {'extraction': {'Smoothness': 82.3, 'Integrity': 92.7, 'Accuracy': 98.1}, 
            'gan':        {'Smoothness': 9.5,  'Integrity': 13.6, 'Accuracy': 4.2}}
rf_accuracy = 87.5
current_path = os.path.dirname(__file__)

def generate_feedback(num:int=100, sigma:int=10):
    """
    Function
        Generate reader feedbacks on two summaries on Smoothness, Integrity and Accuracy
    Input
        num: number of reader feedbacks
        sigma: normal distribution variance
    """
    scores = {'extraction': {'Smoothness': [], 'Integrity': [], 'Accuracy': []}, 
              'gan':        {'Smoothness': [], 'Integrity': [], 'Accuracy': []}}
    
    
    for i in range(num):
        for method in scores:
            for aspect in ['Smoothness', 'Integrity', 'Accuracy']:
                r = np.random.normal(standard[method][aspect],sigma)
                if r < 0: scores[method][aspect].append(0)
                elif r > 100: scores[method][aspect].append(100)
                else: scores[method][aspect].append(r)
    
    with open(current_path + "\\scores.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Smoothness', 'Integrity', 'Accuracy','Smoothness', 'Integrity', 'Accuracy'])
        for i in range(num):
            writer.writerow([scores['extraction']['Smoothness'][i],scores['extraction']['Integrity'][i],scores['extraction']['Accuracy'][i],
                             scores['gan']['Smoothness'][i],scores['gan']['Integrity'][i],scores['gan']['Accuracy'][i]])
    """
    for aspect in ['Smoothness', 'Integrity', 'Accuracy']:
        line1 = plt.scatter(range(num), scores['extraction'][aspect], color='red')
        line2 = plt.scatter(range(num), scores['gan'][aspect], color='blue')
        plt.legend([line1, line2], ["Extraction", "Gan"], loc=1)
        plt.xlabel('Readers')
        plt.ylabel('Rating on ' + aspect)
        plt.title('Reader Feedbacks on ' + aspect)
        plt.show()
    """
    
def generate_rf_accuracy(num:int=100, sigma:int=5):
    """
    Function
        Generate accuracy of random forest model
    Input
        num: number of reader feedbacks
        sigma: normal distribution variance
    """
    samples = []
    for i in range(num):
        r = np.random.normal(rf_accuracy,sigma)
        if r < 0: samples.append(0)
        elif r > 100: samples.append(100)
        else: samples.append(r)
    
    line1 = plt.scatter(range(num), samples, color='red')
    plt.legend([line1], ["Accuracy"], loc=1)
    plt.xlabel('Number of Tests')
    plt.ylabel('Accuracy in Percentage')
    plt.title('Random Forest Sentiment Classifier Accuracy')
    plt.show()
if __name__ == "__main__":
    generate_feedback()
    #generate_rf_accuracy()