
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
len({word: None for word in text.split()})
x = text[:100]
print(len(x.split()))
```

    15
    


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.251908396946565
    Number of lines: 4258
    Average number of words in each line: 11.50164396430249
    
    The sentences 0 to 10:
    
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    

## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    unique_words = set(text)
    ids = range(0,len(unique_words))
    vocab_to_int = dict(zip(unique_words, ids))
    int_to_vocab = dict(zip(ids, unique_words))
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Couldn't import dot_parser, loading of dot files will not be possible.
    Tests Passed
    


```python
a,b = create_lookup_tables(text[:100].split())
print(a, b)
```

    {'Where': 0, 'elite': 1, 'PHONE)': 2, 'meet': 3, 'to': 4, 'the': 5, 'Bart_Simpson:': 6, 'hell': 9, 'yeah,': 7, '(INTO': 8, 'Moe_Szyslak:': 10, 'Eh,': 12, 'Tavern.': 13, "Moe's": 14, 'drink.': 11} {0: 'Where', 1: 'elite', 2: 'PHONE)', 3: 'meet', 4: 'to', 5: 'the', 6: 'Bart_Simpson:', 7: 'yeah,', 8: '(INTO', 9: 'hell', 10: 'Moe_Szyslak:', 11: 'drink.', 12: 'Eh,', 13: 'Tavern.', 14: "Moe's"}
    

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    token = {'--': '||Dash||', '?': '||Question_mark||', ')': '||Right_Parentheses||', '.': '||Period||',
             '\n': '||Return||', '!': '||Exclamation_mark||', '"': '||Quotation_Mark||', ';': '||Semicolon||',
             ',': '||Comma||', '(': '||Left_Parentheses||'}
    return token

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed
    

## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.1.0
    

    C:\Users\Kaustubh\Anaconda3\envs\DLND\lib\site-packages\ipykernel\__main__.py:14: UserWarning: No GPU found. Please use a GPU to train your neural network.
    

### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    Input = tf.placeholder(tf.int32, [None, None], name='input')
    Targets = tf.placeholder(tf.int32, [None, None], name='targets')
    LearningRate = tf.placeholder(tf.float32, name='lr')
    return (Input, Targets, LearningRate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed
    

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
keep_prob = 0.8
num_layers = 2

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    
    # Add dropout to the cell
    #drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    #cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)], state_is_tuple=True)
    
    #modified for tf 1.1.0
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_size), output_keep_prob=keep_prob) for _ in range(num_layers)])

    #cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name= "initial_state")
    return cell, initial_state

"""
stacked_lstm = tf.contrib.rnn.MultiRNNCell([build_rnn_cell(rnn_size) for _ in range(num_of_layers)], state_is_tuple=True)

def build_rnn_cell(rnn_size):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True) 
    drop = tf.contrib.rnn.DropoutWrapper(lstmCell, input_keep_prob=1.0, output_keep_prob=0.5)
    return drop
"""
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

tests.test_get_init_cell(get_init_cell)
```

    Tests Passed
    

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.contrib.layers.embed_sequence(
                ids=input_data,
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                unique=False,
                initializer=None,
                regularizer=None,
                trainable=True,
                scope=None,
                reuse=None
                )
    return embedding


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed
    

### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    Outputs, Final_State=tf.nn.dynamic_rnn(
        cell,
        inputs,
        #sequence_length=None,
        #initial_state=None,
        dtype=tf.float32
        #parallel_iterations=None,
        #swap_memory=False,
        #time_major=False,
        #scope=reuse
        )
    Final_State=tf.identity(Final_State, 'final_state')
    return Outputs, Final_State


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed
    

### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    input_embeddings=get_embed(input_data, vocab_size, embed_dim)
    rnn_outputs, Final_State=build_rnn(cell, input_embeddings)
    Logits=tf.contrib.layers.fully_connected(rnn_outputs, vocab_size, None)
    return Logits, Final_State


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed
    

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    bs, sl = batch_size, seq_length
    
    #num of sequences
    a=int((len(int_text)/sl)-1)
    
    #num of batches
    b=int(a/bs)
    
    op=np.zeros([b, 2, bs, sl], dtype=np.int32)
    
    l=0;
    
    for i in range(bs):
        for j in range(b):
            op[j][0][i]=int_text[l:l+sl]
            op[j][1][i]=int_text[l+1:l+sl+1]
            l=l+sl
    #print(op)
    
    op[j][1][i][sl-1]=int_text[0]
    Batches=op
    return Batches


#print(get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2))
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

tests.test_get_batches(get_batches)
```

    Tests Passed
    

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 320
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 256
# Sequence Length
seq_length = 12
# Learning Rate
learning_rate = 0.002
# Show stats for every n number of batches
show_every_n_batches = 105

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/22   train_loss = 8.822
    Epoch   4 Batch   17/22   train_loss = 6.090
    Epoch   9 Batch   12/22   train_loss = 5.998
    Epoch  14 Batch    7/22   train_loss = 6.091
    Epoch  19 Batch    2/22   train_loss = 5.666
    Epoch  23 Batch   19/22   train_loss = 5.705
    Epoch  28 Batch   14/22   train_loss = 5.519
    Epoch  33 Batch    9/22   train_loss = 5.279
    Epoch  38 Batch    4/22   train_loss = 4.845
    Epoch  42 Batch   21/22   train_loss = 4.735
    Epoch  47 Batch   16/22   train_loss = 4.479
    Epoch  52 Batch   11/22   train_loss = 4.399
    Epoch  57 Batch    6/22   train_loss = 4.125
    Epoch  62 Batch    1/22   train_loss = 4.010
    Epoch  66 Batch   18/22   train_loss = 3.926
    Epoch  71 Batch   13/22   train_loss = 3.650
    Epoch  76 Batch    8/22   train_loss = 3.584
    Epoch  81 Batch    3/22   train_loss = 3.463
    Epoch  85 Batch   20/22   train_loss = 3.283
    Epoch  90 Batch   15/22   train_loss = 3.256
    Epoch  95 Batch   10/22   train_loss = 3.091
    Epoch 100 Batch    5/22   train_loss = 2.991
    Epoch 105 Batch    0/22   train_loss = 2.900
    Epoch 109 Batch   17/22   train_loss = 2.787
    Epoch 114 Batch   12/22   train_loss = 2.748
    Epoch 119 Batch    7/22   train_loss = 2.655
    Epoch 124 Batch    2/22   train_loss = 2.601
    Epoch 128 Batch   19/22   train_loss = 2.516
    Epoch 133 Batch   14/22   train_loss = 2.451
    Epoch 138 Batch    9/22   train_loss = 2.374
    Epoch 143 Batch    4/22   train_loss = 2.225
    Epoch 147 Batch   21/22   train_loss = 2.065
    Epoch 152 Batch   16/22   train_loss = 2.124
    Epoch 157 Batch   11/22   train_loss = 2.159
    Epoch 162 Batch    6/22   train_loss = 1.973
    Epoch 167 Batch    1/22   train_loss = 1.944
    Epoch 171 Batch   18/22   train_loss = 1.988
    Epoch 176 Batch   13/22   train_loss = 1.853
    Epoch 181 Batch    8/22   train_loss = 1.777
    Epoch 186 Batch    3/22   train_loss = 1.826
    Epoch 190 Batch   20/22   train_loss = 1.705
    Epoch 195 Batch   15/22   train_loss = 1.768
    Epoch 200 Batch   10/22   train_loss = 1.655
    Epoch 205 Batch    5/22   train_loss = 1.619
    Epoch 210 Batch    0/22   train_loss = 1.579
    Epoch 214 Batch   17/22   train_loss = 1.516
    Epoch 219 Batch   12/22   train_loss = 1.520
    Epoch 224 Batch    7/22   train_loss = 1.382
    Epoch 229 Batch    2/22   train_loss = 1.396
    Epoch 233 Batch   19/22   train_loss = 1.357
    Epoch 238 Batch   14/22   train_loss = 1.394
    Epoch 243 Batch    9/22   train_loss = 1.332
    Epoch 248 Batch    4/22   train_loss = 1.236
    Epoch 252 Batch   21/22   train_loss = 1.131
    Epoch 257 Batch   16/22   train_loss = 1.201
    Epoch 262 Batch   11/22   train_loss = 1.224
    Epoch 267 Batch    6/22   train_loss = 1.117
    Epoch 272 Batch    1/22   train_loss = 1.097
    Epoch 276 Batch   18/22   train_loss = 1.113
    Epoch 281 Batch   13/22   train_loss = 1.085
    Epoch 286 Batch    8/22   train_loss = 1.025
    Epoch 291 Batch    3/22   train_loss = 1.078
    Epoch 295 Batch   20/22   train_loss = 1.010
    Epoch 300 Batch   15/22   train_loss = 1.051
    Epoch 305 Batch   10/22   train_loss = 1.020
    Epoch 310 Batch    5/22   train_loss = 1.004
    Epoch 315 Batch    0/22   train_loss = 1.024
    Epoch 319 Batch   17/22   train_loss = 0.938
    Model Trained and Saved
    

## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    InputTensor = loaded_graph.get_tensor_by_name('input:0')
    InitialStateTensor = loaded_graph.get_tensor_by_name('initial_state:0')
    FinalStateTensor = loaded_graph.get_tensor_by_name('final_state:0')
    ProbsTensor = loaded_graph.get_tensor_by_name('probs:0')
    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed
    

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    picked_word=np.random.choice(list(int_to_vocab.values()), None, replace=False, p=probabilities)
    return picked_word


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed
    

## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    INFO:tensorflow:Restoring parameters from ./save
    moe_szyslak: okay lisa_simpson: last one way!
    carl_carlson: homer, thanks, moe. i had a feeling those two more day, i aren't that remembered.
    lenny_leonard: you were nothin', i'm really is about fire on?
    homer_simpson: well, if you think believe there you will ain't 'cause i don't wanna do you through?" you've sure might edna, i'm drunk again, three, and they can close your job. there says for me yet, he is just sure is one of most the ladies like my name is good out. his name"" smells is and i lost left, and does on the middle of moe, he doesn't have doing that enough of the work. it is.
    carl_carlson: what happened today? you've got a lotta brockman of him out?
    carl_carlson: all what it was drinkin' of a sign like my generous see spinning.., uh, i was still wouldn't knew my song.
    homer_simpson: i have left during the half back on that.
    homer_simpson: nice for someone. have a guy with
    

# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
