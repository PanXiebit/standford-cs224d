tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
print(tagnames[2:10])
# print(list(enumerate(tagnames)))
# print(dict(enumerate(tagnames)))
#
# from data_utils.utils import *
# import data_utils.ner as ner
# tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
# num_to_tag = dict(enumerate(tagnames)) # {0: 'O', 1: 'LOC', 2: 'MISC', 3: 'ORG', 4: 'PER'}
# tag_to_num = {v:k for k,v in num_to_tag.items()} # invert dict
#
#
#
# docs = load_dataset('/home/panxie/Documents/cs224d/assignment2_release/data/ner/train')
#
# wv, word_to_num, num_to_word = ner.load_wv('/home/panxie/Documents/cs224d/assignment2_release/data/ner/vocab.txt',
#                                            '/home/panxie/Documents/cs224d/assignment2_release/data/ner/wordVectors.txt')
#
# X_train, y_train = docs_to_windows(
#         docs, word_to_num, tag_to_num, wsize=3)
#
# print(len(X_train))
# print(X_train[0])
