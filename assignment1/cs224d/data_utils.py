#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            path = "cs224d/datasets/stanfordSentimentTreebank"

        self.path = path
        self.tablesize = tablesize


    # 从stanfordSentimentTreebank中获取词典
    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx   # 用dict表示的词典 {'the': 0, 'rock': 1, 'is': 2, 'destined': 3, 'to': 4,...}
                    revtokens += [w]  # 列表，存放word ['the', 'rock', 'is', 'destined', 'to', 'be', '21st', ....]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1  # 语料库中出现词的频率，或许在huffman树中有用{'the': 10128, 'rock': 34, 'is': 3558, 'destined': 8, 'to': 4234, ...}

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens


    # 从datasetSentences.txt中获取sentences
    def sentences(self):
        # getattr(obj, name) 判断obj对象是否包含属性name
        # 如果有这个属性，那么引用sentences这个函数时就返回 self._sentence
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower() for w in splitted]]
                
        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    # snetences的个数
    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences()) # sentences是list
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences() # list
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[w for w in s if 0 >= rejectProb[tokens[w]]
                         or random.random() >= rejectProb[tokens[w]]]for s in sentences * 30]  # 从所有的30倍的sentences随机选择

        allsentences = [s for s in allsentences if len(s) > 1] # 选出长度大于1的
        
        self._allsentences = allsentences
        
        return self._allsentences

    def getRandomContext(self, C=5):  # 随机选取上下文
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)  # 句子id
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1) # 对应句子中选取中心词的id

        context = sent[max(0, wordID - C):wordID] # 如果id小于5，也就是前文没有C个词，那就从0开始呗
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)] # 后文也是一样的，要判断时候够C个

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def sent_labels(self):   # 每个sentence对应的标签
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in range(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            try:
                sent_labels[i] = labels[dictionary[full_sent]]
            except KeyError:
                continue
                # print('KeyError')
                # print(full_sent)
                # print(full_sent.encode("uft-8"))

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in range(3)] #　[[],[],[]] 分别表示训练集，验证集和测试集
        with open(self.path + "/datasetSplit.txt", "r") as f: # sentence_index,splitset_label
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]  # 将已经分类好的数据分别放到train,dev,test中

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label): # 根据sentiment的值将sentences分为4类
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split() # [[],[],[]]
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount  # 阈值，2.27246

        nTokens = len(self.tokens()) # 19539
        rejectProb = np.zeros((nTokens,))  # (19539,)对词典中每个词赋予一个概率
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq)) # 出现次数小于2.272的词概率为0，频率越大概率越大

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]