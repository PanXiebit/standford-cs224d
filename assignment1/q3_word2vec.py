import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x -= np.mean(x, axis=1, keepdims=True)
    x /= np.std(x, axis=1, keepdims=True)
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    V ,D = outputVectors.shape # V表示词表大小， D表示词向量的维度
    assert predicted.shape == (1,D) # predicted 表示从inputVectors中提取的词向量

    # score
    score = np.sum(outputVectors * predicted, axis=1) # score.shape = (V,1)
    # probability y
    prob = softmax(score)
    # cross-entropy
    cost = -np.log(prob[target])

    # gradient, derivative on the score
    dscore = prob.copy()
    dscore[target] -= 1.0    # error: e_j = (y_j-t_j)

    # derivative on the outputVectors 其实这里不用reshape也可以，本身shape就符合
    grad = np.dot(dscore.reshape((V,1)), predicted.reshape((1,D))) # VxD

    # derivative on the predicted
    # \sum_{j=1}^Ve_jW_{ij} = \sum_{j=1}^V (y_j-t_j)W_{ij}
    gradPred = np.sum(outputVectors * prob.reshape((V, 1)), axis=0) - outputVectors[target]  # (1, D)
    ### END YOUR CODE
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models

    """
    
    ### YOUR CODE HERE
    V, D = outputVectors.shape

    # get the k random indices
    k_indicies = []
    for i in range(K):
        rand_index = dataset.sampleTokenIdx() # 有没有可能随机到正确的样本？有，但概率很小
        k_indicies.append(rand_index)

    # loss function
    neg_sample_vector = outputVectors[k_indicies, :] # KxD
    assert neg_sample_vector.shape == (K,D)
    sigm_neg = sigmoid(-1.0 * np.dot(neg_sample_vector, predicted.reshape((D,1)))) # KxD Dx1 = Kx1
    cost_neg = np.sum(np.log(sigm_neg), axis=0)

    sigm_cor = sigmoid(np.dot(outputVectors[target], predicted.reshape((D,1))))
    cost = -1.0 * np.log(sigm_cor)- cost_neg

    # gradient on output vectors
    grad = np.zeros(outputVectors.shape) # V, D
    grad[target] = predicted * (sigm_cor - 1.0)  # 1xD
    for k in k_indicies:
        grad[k, :] += -1.0 * predicted.reshape((D,)) * (sigmoid(np.dot(-1.0 * predicted, outputVectors[k])) - 1.0)

    # gradient on input vector
    # 这里第一项减一跟公式有点不一样啊。。但是结果是对的。。1 - sigm_neg.reshape((1,K))
    gradPred_neg = np.dot(1 - sigm_neg.reshape((1,K)), neg_sample_vector).reshape((1, D)) # 1xK KxD = 1xD
    gradPred_cor = (sigm_cor-1) * outputVectors[target].reshape((1, D))
    gradPred = gradPred_neg + gradPred_cor

    ### END YOUR CODE
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    # 根据中心词预测上下文

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    V, D = inputVectors.shape # N表示词表大小？D表示词向量的维度？

    curr_index = tokens[currentWord] # word to index, integer
    curr_vector = inputVectors[curr_index].reshape(1, D) # (1,D) 这里需要注意(D,)和(1,D)的区别
    assert curr_vector.shape == (1, D)

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    for context_word in contextWords:
        context_index = tokens[context_word]
        cost_curr, grad_in_curr, grad_out_curr = word2vecCostAndGradient(curr_vector,context_index,outputVectors,dataset)
        cost += cost_curr
        gradIn[curr_index, :] += grad_in_curr.reshape((D,)) # only update current word vector
        gradOut += grad_out_curr
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    # from centre word generate context words
    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    V, D = outputVectors.shape
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    curr_index = tokens[currentWord] #centre word

    context_index = [tokens[context_word] for context_word in contextWords]
    context_vectors = inputVectors[context_index]   # shape=(N,D)   N means context words number
    assert context_vectors.shape == (len(context_index), D)
    context_vectors_sum = np.sum(context_vectors, axis=0).reshape((1,D))   ## 这里跟公式也不太一样，没有求平均值，但结果是对的。。。

    cost, grad_in_curr, grad_out_curr = word2vecCostAndGradient(context_vectors_sum, curr_index, outputVectors, dataset)
    for c in context_index:
        gradIn[c, :] += grad_in_curr.reshape((D,)) # every context word share the same gradient???
    gradOut += grad_out_curr
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    """

    :param word2vecModel: skip-gram or CBOW
    :param tokens: a map that word to its index
    :param wordVectors: shape=(2N, D), concatenated by inputVectors and outputVectors , N means vocabulary size
    :param dataset:
    :param C: context words number is 2C
    :param word2vecCostAndGradient: default softmaxCostAndGradient
    :return:
    """
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:int(N/2),:]
    outputVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1) # context words的数量是随机的？？？
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})() # 实例化一个类， 用来测试～
    def dummySampleTokenIdx():
        return random.randint(0, 4) # [0,4]包括两个端点

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)]
                                             for i in range(2*C)]  # 随机生成一个中心词和2C个上下文词('c', ['e', 'e', 'd', 'e'])
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3)) # 向量初始化, 2N=10, D=3
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("\n")
    print("==== Gradient check for skip-gram and softmax ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors) # 默认使用word2vecCostAndGradient
    print("==== Gradient check for skip-gram and negative sampling ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print("\n==== Gradient check for CBOW and softmax ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    print("\n==== Gradient check for CBOW and negative sampling ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()