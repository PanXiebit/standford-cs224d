import random
import numpy as np
from cs224d.data_utils import *
import matplotlib.pyplot as plt

from q3_word2vec import *
from q3_sgd import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens) #　19539

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / \
							  dimVectors, np.zeros((nWords, dimVectors))), axis=0)
print("wordVectors.shape = {0}".format(wordVectors.shape))

# 使用sgd训练词向量, wordVectors0是inputvectors和outputvectors的叠加
wordVectors0 = sgd(
    f=lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
    	negSamplingCostAndGradient), x0=wordVectors, step=0.3,
		iterations=40000, postprocessing=False, useSaved=True, PRINT_EVERY=100)

print ("sanity check: cost at convergence should be around or below 10")

# sum the input and output word vectors
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

# Visualize the word vectors you trained
_, wordVectors0, _ = load_saved_params()
# 这里是直接将inputvectors和outputvectors相加
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", 
	"good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
	"worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", 
	"annoying"]
visualizeIdx = [tokens[word] for word in visualizeWords]

# 可视化
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2]) 

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], 
    	bbox=dict(facecolor='green', alpha=0.1))
    
plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')
plt.show()