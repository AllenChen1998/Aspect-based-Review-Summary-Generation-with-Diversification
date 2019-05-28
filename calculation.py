import os
import pickle
import numpy as np
from helpers import *

"""
Parameters
    model: gensim object
        Gensim embedding matrix
IMPORTANT
    embedding_matrix.pickle
    should be placed with this py file
"""

current_path = os.path.dirname(__file__)
model = pickle.load(open(current_path + "\\embedding_matrix.pickle", "rb"))


def cal_sentence_vector(s:str):
    """
    Function
        Calculate sentence vector by averaging its word vectors
    Input
        s: target sentence
    Output
        sentence_vector: numpy vectors
    """
    sentence = cut(s)
    vector = np.zeros((300,))
    for word in sentence:
        if word not in model.wv: continue
        vector += model.wv[word]
    return vector/len(sentence)


def cal_similarity_matrix(ss:list):
    """
    Function
        Calculate similarity matrix of all sentences by cosine distance
    Input
        ss: target sentences
    Output
        s_mat: np.array similarity matrix
    """
    n = len(ss)
    s_mat = np.zeros((n, n))
    vectors = np.zeros((n, 300))
    
    for i in range(n):
        vectors[i] = cal_sentence_vector(ss[i])
    
    lengths = np.linalg.norm(vectors, axis=1).reshape(n, -1)
    lengths_product = lengths.dot(lengths.T)
    return 0.5 + 0.5 * vectors.dot(vectors.T)/lengths_product


def cal_transfer_matrix(ss:list):
    """
    Function
        Calculate transition matrix of all sentences
        Transfer probability from a to b is proportional to 1 - cal_similarity(a, b)
    Input
        ss: target sentences
    Output
        s_mat_r: np.array transition matrix
    """
    n = len(ss)
    s_mat_r = np.ones((n, n))*1.0001 - cal_similarity_matrix(ss)
    for i in range(n):
        s_mat_r[i] /= np.sum(s_mat_r[i])
    return s_mat_r


def sample(ss:list, num_of_sample:int):
    """
    Function
        Sample num_of_sample sentences according to transition matrix
    Input
        ss: target sentences
        num_of_sample: number of samples
    Output
        samples: list of samples
    """
    n = len(ss)
    if num_of_sample > n:
        raise RuntimeError('SampleTooMuchError')
    
    print('计算 %d x %d 转移矩阵'%(n, n))
    t_mat = cal_transfer_matrix(ss)
    
    samples = []
    sampled = []
    index = 0
    
    print("采样成功 ",end=' ')
    while len(samples) < num_of_sample:
        if index not in sampled:
            samples.append(ss[index])
            sampled.append(index)
            print(len(samples), end=' ')
        index = np.random.choice(np.array(range(n)),p=t_mat[index])
    print()
    return samples