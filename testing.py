from imutils import paths
import numpy as np
import random
import cv2
import os
import re
import nltk
from gensim.sklearn_api import W2VTransformer
from gensim.models import Word2Vec, word2vec
import pickle
from Crypto.Cipher import AES
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
BS = 16
pad = lambda s: bytes(s + (BS - len(s) % BS) * chr(BS - len(s) % BS), 'utf-8')
unpad = lambda s : s[0:-ord(s[-1:])]
class AESCipher:

    def __init__( self, key ):
        self.key = bytes(key, 'utf-8')

    def encrypt( self, raw ):
        raw = pad(raw)
        iv = "encryptionIntVec".encode('utf-8')
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return base64.b64encode(cipher.encrypt( raw ) )
    def decrypt( self, enc ):
        iv = "encryptionIntVec".encode('utf-8')
        enc = base64.b64decode(enc)
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return unpad(cipher.decrypt( enc )).decode('utf8')
cipher = AESCipher('enIntVecTest2020')


##decrypted = cipher.decrypt(encrypted)
##print(decrypted)


def testing():
    model_name = 'train_model'
    # Set values for various word2vec parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 1   # Minimum word count                        
    num_workers = 3       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    model = Word2Vec.load(model_name)

    def make_feature_vec(words, model, num_features):
        """
        Average the word vectors for a set of words
        """
        feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
        nwords = 0.
        index2word_set = set(model.wv.index2word)  # words known to the model

        for word in words:
            if word in index2word_set: 
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec,model[word])
    ##    print(feature_vec)
    ##    print(nwords)
        feature_vec = np.divide(np.array(feature_vec), int(nwords))
        return feature_vec


    def get_avg_feature_vecs(reviews, model, num_features):
        """
        Calculate average feature vectors for all reviews
        """
        counter = 0.
        review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')  # pre-initialize (for speed)
        
        for review in reviews:
            review_feature_vecs[int(counter)] = make_feature_vec(review, model, num_features)
            counter = counter + 1.
        return review_feature_vecs

    with open(r'dataset\fraud\CheckupActivity.java','r') as file:
        xdata = file.read()
##    encrypted = cipher.encrypt(xdata)
##    print(type(encrypted))
##    # Write-Overwrites
##    file1 = open("aesdata.txt","wb")#write mode
##    file1.write(encrypted)
##    file1.close()



                            
    # Cleaing the text
    processed_article = xdata.lower()
    processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
    processed_article = re.sub(r'\s+', ' ', processed_article)

    # Preparing the dataset
    all_sentences = nltk.sent_tokenize(processed_article)

    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

    # Removing Stop Words
    from nltk.corpus import stopwords
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]
    print(all_words[0])
    test=all_words
    dataVecs = get_avg_feature_vecs(test, model, num_features)
    print(dataVecs.shape)
    Xtest=np.nan_to_num(dataVecs)
    print(Xtest.shape)
    f=open("svm_model.pickle",'rb')
    svm=pickle.load(f)
    f.close()
    y=svm.predict(Xtest.reshape(1,-1))
    print(y)
testing()
