# Create a classification dataset for the movie reviews
import csv
import os, random, sys, copy
import torch, torch.nn as nn, numpy as np
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

#ADOPTED FROM MOVIE DATALOADER FROM HW3
class sarcasmDataset(torch.utils.data.Dataset):

    def __init__(self, input_file="..\\train.En.csv", word2id=None, finalized_data = None, data_limit=1000, max_length=280):
        """
        :param directory: The location of aclImdb
        :param split: Train or test
        :param word2id: The generated glove word2id dictionary
        :param finalized_data: We'll use this to initialize a validation set without reloading the data.
        :param data_limit: Limiter on the number of examples we load
        :param max_length: Maximum length of the sequence
        """

        if finalized_data:
            self.data = finalized_data
        else:
            #dataset parameters
            self.data_limit = data_limit
            self.max_length = max_length
            self.word2id = word2id

            #read the input file and get examples: example = (tweet_text, sarcasm_label)
            examples = self.read_file(input_file)

            #get tokenized examples: example_tokenized = (tweet_text_embeddings, sarcasm_label)
            examples_tokenized = self.tokenize(examples) 

            #set dataset data and shuffle it
            self.data = examples_tokenized
        random.shuffle(self.data)

    def read_file(self, input_file):
        examples = []
        with open(input_file, 'r', errors='ignore') as csvfile: #read the csv
            reader = csv.reader(csvfile)
            next(reader) #skip the header
            for line in reader: #go through every tweet example
                examples.append([line[1],int(line[2])]) #get the tweet and the true sarcasm label
        return examples

    def tokenize(self, examples):

        example_ids = []
        misses = 0              # Count the number of tokens in our dataset which are not covered by glove -- i.e. percentage of unk tokens
        total = 0
        for example in examples: #for every example
            tokens = word_tokenize(example[0]) #tokenize the tweet_text
            ids = []
            for tok in tokens: #go through every word in tokenized tweet and get embedding from glove
                if tok in word2id: 
                    ids.append(word2id[tok])
                else:
                    misses += 1
                    ids.append(word2id['unk'])
                total += 1

            if len(ids) >= self.max_length:
                ids = ids[:self.max_length]
                length = self.max_length
            else:
                length = len(ids)
                ids = ids + [word2id['<pad>']]*(self.max_length - len(ids))
            if length > 0:
                example_ids.append(((torch.tensor(ids),length),example[1]))
        print('Missed {} out of {} words -- {:.2f}%'.format(misses, total, misses/total))
        return (example_ids)

    def generate_validation_split(self, ratio=0.8):

        split_idx = int(ratio * len(self.data))

        # Take a chunk of the processed data, and return it in order to initialize a validation dataset
        validation_split = self.data[split_idx:]

        #We'll remove this data from the training data to prevent leakage
        self.data = self.data[:split_idx]

        return validation_split

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    
    glove_file = '../glove.6B.50d.txt'

    embeddings_dict = {}

    with open(glove_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i == 0:
                print(line)
            line = line.strip().split(' ')
            word = line[0]
            embed = np.asarray(line[1:], "float")

            embeddings_dict[word] = embed

    print('Loaded {} words from glove'.format(len(embeddings_dict)))

    embedding_matrix = np.zeros((len(embeddings_dict)+1, 50)) #add 1 for padding

    word2id = {}
    for i, word in enumerate(embeddings_dict.keys()):

        word2id[word] = i                                #Map each word to an index
        embedding_matrix[i] = embeddings_dict[word]      #That index holds the Glove embedding in the embedding matrix
    word2id['<pad>'] = 0
    train_dataset = sarcasmDataset(word2id=word2id)
    valid_data = train_dataset.generate_validation_split()
    valid_dataset = sarcasmDataset(finalized_data=valid_data, word2id=word2id)
    print(train_dataset.__len__(), train_dataset[1])
    print(valid_dataset.__len__(), valid_dataset[1])