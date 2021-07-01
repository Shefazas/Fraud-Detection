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
print("[INFO] loading datas...")
data = []
labels = []

path ="dataset"
#we shall store all the file names in this list
filelist = []

for root, dirs, files in os.walk(path):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file))

for name in filelist:
    with open(name,'r') as file:
        xdata = file.read()
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
    data.append(all_words[0])
    lab=name.split(os.path.sep)[-2]
    if (lab=="fraud"):
        labels.append(0)
    else:
        labels.append(1)
print(len(data))
model_name = 'train_model'
print('---------------------------------------')
# Set values for various word2vec parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count                        
num_workers = 3       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
##vocab = [s.encode('utf-8').split() for s in sentences]

model = word2vec.Word2Vec(data, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
##print(model.init_sims(replace=True))
model.save(model_name)
##print(model.most_similar("know"))
##print(model.most_similar(positive=[ 'was'], negative=['has']))

def make_feature_vec(words, model, num_features):
    import numpy as np
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
    counter = 0.
    import numpy as np
    review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')  # pre-initialize (for speed)
    
    for review in reviews:
        review_feature_vecs[int(counter)] = make_feature_vec(review, model, num_features)
        counter = counter + 1.
    return review_feature_vecs

##clean_train_reviews = []
import numpy as np
##for review in train_reviews['Text']:
##    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = get_avg_feature_vecs(data, model, num_features)
print(trainDataVecs[3],labels[3])
# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(trainDataVecs, labels, test_size=0.05,random_state=109) # 70% training and 30% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)
print(X_test[0].shape)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(y_pred)
f=open("svm_model.pickle","wb")
pickle.dump(clf,f)
f.close
print("model saved")
