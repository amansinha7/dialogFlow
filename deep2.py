from __future__ import print_function

import collections
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import nltk
import numpy as np
from flask import Flask, make_response, request
from future.standard_library import install_aliases
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

install_aliases()




# Flask app should start in global layout
app = Flask(__name__)

@app.route('/dialogflow', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))
    if req.get("result").get("action") != "logSR":
        return {}
    
    np.random.seed(42)

    INPUT_FILE = "data/training2.txt"

    VOCAB_SIZE = 5000
    EMBED_SIZE = 100
    NUM_FILTERS = 256
    NUM_WORDS = 3
    BATCH_SIZE = 64
    NUM_EPOCHS = 20

    counter = collections.Counter()
    fin = open(INPUT_FILE, "r")
    maxlen = 0
    i=1
    for line in fin:
        _, sent = line.strip().split("\t")
        words = [x.lower() for x in nltk.word_tokenize(sent)]
        #print("i=",i)
        #i=i+1
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1
    fin.close()

    word2index = collections.defaultdict(int)
    for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
        word2index[word[0]] = wid + 1
    vocab_sz = len(word2index) + 1
    index2word = {v:k for k, v in word2index.items()}

    xs, ys = [], []
    fin = open(INPUT_FILE, "r")
    for line in fin:
        label, sent = line.strip().split("\t")
        ys.append(int(label))
        words = [x.lower() for x in nltk.word_tokenize(sent)]
        wids = [word2index[word] for word in words]
        xs.append(wids)
    fin.close()
    X = pad_sequences(xs, maxlen=maxlen)
    Y = np_utils.to_categorical(ys)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=42)
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

    model = Sequential()
    model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen))
    #model.add(SpatialDropout1D(Dropout(0.2)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(3, activation="softmax")) # AKP: 3 is number of bits needed (001,010)
    print("Xtrain=",Xtrain)
    print("Ytrain=",Ytrain)
    print("Xtest=",Xtest)
    print("Ytest=",Ytest)

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                metrics=["accuracy"])
    history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=(Xtest, Ytest)) 
                        
    # plt.subplot(211)
    # plt.title("accuracy")
    # plt.plot(history.history["acc"], color="r", label="train")
    # plt.plot(history.history["val_acc"], color="b", label="validation")
    # plt.legend(loc="best")

    # plt.subplot(212)
    # plt.title("loss")
    # plt.plot(history.history["loss"], color="r", label="train")
    # plt.plot(history.history["val_loss"], color="b", label="validation")
    # plt.legend(loc="best")

    # plt.tight_layout()
    # plt.show()

    score = model.evaluate(Xtest, Ytest, verbose=1)
    print("Test score: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))

    xs1, ys1 = [], []
    #line1="0 ,Advance billing program errors for future GL date invoices"
    #line1="0,Evergreen billing is not generating billing stream when some contract has error"
    result = req.get("result")
    parameters = result.get("parameters")
    text = parameters.get("text")

    line1="0," + text
    #Todo: line1 = "0,<<user input>>"

    label1, sent1 = line1.strip().split(",")
    ys1.append(int(label1))
    words1 = [x.lower() for x in nltk.word_tokenize(sent1)]
    wids1 = [word2index[word] for word in words1]
    xs1.append(wids1)
    X1 = pad_sequences(xs1, maxlen=maxlen)
    Y1 = np_utils.to_categorical(ys1)

    #X1=["Stream billing is not working"]
    y1=model.predict(X1)
    print("y1=",y1)

    #Todo: Scan through y1[0] which is again a list and see which index is having the highest probability. Return that index to the client.
    max = 0
    index = -1
    index2 = -1
    for index,value in enumerate(y1[0]):
        print(index,value)
        if max < value:
            max = value
            index2 = index
    print('Max ', max,' index2 ',index2)
    SOLUTION_FILE = "data/keys.txt"
    fin2 = open(SOLUTION_FILE, "r")
    i=0
    dict={}
    for line in fin2:
        i = i + 1
        code, ID, bug, solution = line.strip().split(",")
        list=[code, bug, solution]
        dict[ID]= list

    NUMRECS = i
    fin2.close()

    sol1=dict.get(str(index2))
    if (sol1 is None) :
        speech = "Could not understand your issue. Please check again later..."
    else:
        speech="Your issue looks similar to " + sol1[0] + " ("+ "bug "+ sol1[1]+ "). " + sol1[2] + ". Do you still want to log an SR?"

    print("answer=",speech)

    #speech = "Issue " + str(index2) + " encountered"
    res = {"speech": speech,
    "displayText": speech,
    "source": "service-request-dialogflow"}

    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

if __name__ == '__main__':
	port = int(os.getenv('PORT', 5000))
	print("Starting app on port %d" % port)
	app.run(debug=False, port=port, host='0.0.0.0')
    
