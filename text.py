import numpy
import sys
import nltk
nltk.download('stopwords')
from nick.tokenize import RegexpTokenizer
from nick.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


f = open('frankenstein-2.txt').read()

#tokenization
#standardization
def tokenize_words(input):
    input = input.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokeinzer.tokenize(input)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return ''.join(filtered)

processed_inputs = tokenize_words(f)

#chars to numbers
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i, c in enumerate(chars))

#check if words to chars or chars to num has worked
input_len = len(processed_inputs)
vocab_len = len(chars)
print('Total number of characters: ', input_len)
print('Toatal vocab: ',vocab_len)

#seq length
seq_length = 100
x_data = []
y_data = []

#loop through sequence
for i in range(0, input_len - seq_length, 1):
    in_seq = processed_inputs[i:i+seq_length]
    out_seq = processed_inputs(i+seq_length)
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print('Total Patterns: ',n_patterns)

#convert input seq to np array
X = numpy.reshape(x_data, (n_pattersn,seq_length, 1))
X =X / float(vocab_len)

#one-hot encoding
y = np_utils.to_categorical(y_data)

#Creating the model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y_shape[1], activation = 'softmax'))

#Model compiling
model.compil(loss = 'categorical_crossentropy', optimizer = 'adam')

#Saving weights
filepath = 'model_weights_saved.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
desired_callbacks = [checkpoint]

#fit model and let it train
model.fit(X,y, epochs = 4, batch_size = 256, callbacks = desired_callbacks)

#recompile model along with saved weights
file = 'model_weights_saved.hdf5'
model.load_weights(file)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

#output of model back into characters
num_to_char = dict((i,c) for i,c in enumerate(chars))

#random seed
start = numpy.random.randint(0,len(x_data)-1)
pattern = x_data[start]
print('Random seed')
print('\'',''.join([num_to_char[value] for value in pattern]), '\'')

#generate the text
for i in range(1000):
    x = numpy.reshape(pattern, (1,len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose = 0)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
