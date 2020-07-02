---
title: 'Sentiment Analysis with Transformers'
date: 17-06-2020
header:
  image: "/images/bert_for_sst/transformers_cover.jpg"
---
The Transformer is the latest advance in Deep Learning architectures that has driven most state-of-the-art progress in NLP since it was first presented in ['Attention is All You Need'](https://arxiv.org/abs/1706.03762). Since then, ever and ever [larger](https://arxiv.org/abs/2006.16668) models are being made, with number of parameters shooting into the billions. (side-note: I think we're inflection point in ML with OpenAI's release of their API - everyone now has easy access to these state-of-the-art language models, we're gonna see an explosion of use-cases + value creation as these tools get democratised)

There's a lot of greats resources with visualisations to help understand the architecture which I'll come back to. First, a brief introduction to what makes Transformers so powerful:

*   *Self-attention*: a mechanism allowing us to learn contextual relationships between different elements in our input sequence, replacing the need for sequential structure (from RNN/LSTM cells).
*   *Multi-headed attention*: multiple heads of the model carry out self-attnetion, attending to information jointly at different parts of the sequence from different subspaces. This allows us to learn a variety of features of language + means the model can scale efficiently with large datasets + unsupervised learning.
* *Transfer learning*: Transformers use the knowledge extracted from a prior setting (usually in the form a language model), which can be unsupervised, then apply or *transfer* it to a specific domain, where labelled data is available. This allows a large rich corpus of text to be used in the first pre-training stage, before the model is fine-tuned on custom data. 

In this post, we'll look at how to fine tune a pre-trained model for the task of sentiment analysis using Hugging Face's [Transformer](https://huggingface.co/transformers/pretrained_models.html) library, that gives simple access to many of the top transformed-based models (*BERT*, *GPT-2*, *XLNet* etc).  We'll use [*DistilBert*](https://medium.com/huggingface/distilbert-8cf3380435b5) here, a lightweight version of the famous *BERT* model with 66 million parameters that's slightly easier to run on a single Colab GPU.

BERT stands for Bidirectional Encoder Representations from Transformers. It uses a *masked* language model where 15% of a sequence's tokens are randomly masked, then the model learns to predict, given a token, what came before *or* after it (the bi-dircectional part). In addition, it has a next sentence prediction objective (did this sentence come after a previous one). BERT differs from a more standard *casual* language model, that predicts the most likely next token in the sequence in a left-to-right direction.


## Setup

```python
import transformers
from transformers import DistilBertModel, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from os import path
import requests
import gzip
import zipfile
import numpy as np
from collections import defaultdict

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Loading our Data

For the task of sentiment analysis our model takes a sentence as input and outputs one of five classes representing sentiments (very negative, negative, neutral, positive, very positive). The Stanford Sentiment Treebank (SST-5) is the best-known dataset for this, composed of 11855 such sentences with labels 1-5 already split into train, validation and test sets (of sizes 8544, 1101 and 2210). 

Let's download the dataset, then split into train/val/test sets.

```python
# helper function to download the data
def downloadFile(url,filepath):
    if not path.exists(filepath) :
        with requests.get(url) as r :
            open(filepath, 'wb').write(r.content)
    if not path.exists(filepath[:-4]) :
        with zipfile.ZipFile(filepath,'r') as zp :
            zp.extractall()

# format the data, extracting the sentence as well as the sentiment of the entire sentence
def ReadTextFile(filepath) :
    y = []
    X = []
    with open(filepath) as r :
        for line in r.read().split('\n') :
            #set_trace()
            if len(line)==0 :
                pass
            else :
                y.append(int(line[1]))
                X.append([word[:-1].replace(')','') for word in line.split() if word[-1]==')'])   
    return y, X

downloadFile('https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip', 'trainDevTestTrees_PTB.zip')

y_train, X_train = ReadTextFile("./trees/train.txt")
y_val, X_val = ReadTextFile("./trees/dev.txt")
y_test, X_test = ReadTextFile("./trees/test.txt")
```

We need to turn each sequence of words into tokens that serve as inputs into our model. The `DistilBertTokenizer` object does just that. We can see what the tokenizer does to the first sentence in our training set.

```python
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
```

```python
sample_txt = str(X_train[0])
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')
```

    Sentence: ['The', 'Rock', 'is', 'destined', 'to', 'be', 'the', '21st', 'Century', "'s", 'new', '``', 'Conan', "''", 'and', 'that', 'he', "'s", 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'Arnold', 'Schwarzenegger', ',', 'Jean-Claud', 'Van', 'Damme', 'or', 'Steven', 'Segal', '.']
       
    Tokens: ['[', "'", 'the', "'", ',', "'", 'rock', "'", ',', "'", 'is', "'", ',', "'", 'destined', "'", ',', "'", 'to', "'", ',', "'", 'be', "'", ',', "'", 'the', "'", ',', "'", '21st', "'", ',', "'", 'century', "'", ',', '"', "'", 's', '"', ',', "'", 'new', "'", ',', "'", '`', '`', "'", ',', "'", 'conan', "'", ',', '"', "'", "'", '"', ',', "'", 'and', "'", ',', "'", 'that', "'", ',', "'", 'he', "'", ',', '"', "'", 's', '"', ',', "'", 'going', "'", ',', "'", 'to', "'", ',', "'", 'make', "'", ',', "'", 'a', "'", ',', "'", 'splash', "'", ',', "'", 'even', "'", ',', "'", 'greater', "'", ',', "'", 'than', "'", ',', "'", 'arnold', "'", ',', "'", 'schwarz', '##ene', '##gger', "'", ',', "'", ',', "'", ',', "'", 'jean', '-', 'cl', '##aud', "'", ',', "'", 'van', "'", ',', "'", 'dam', '##me', "'", ',', "'", 'or', "'", ',', "'", 'steven', "'", ',', "'", 'sega', '##l', "'", ',', "'", '.', "'", ']']
    
    Token IDs: [1031, 1005, 1996, 1005, 1010, 1005, 2600, 1005, 1010, 1005, 2003, 1005, 1010, 1005, 16036, 1005, 1010, 1005, 2000, 1005, 1010, 1005, 2022, 1005, 1010, 1005, 1996, 1005, 1010, 1005, 7398, 1005, 1010, 1005, 2301, 1005, 1010, 1000, 1005, 1055, 1000, 1010, 1005, 2047, 1005, 1010, 1005, 1036, 1036, 1005, 1010, 1005, 16608, 1005, 1010, 1000, 1005, 1005, 1000, 1010, 1005, 1998, 1005, 1010, 1005, 2008, 1005, 1010, 1005, 2002, 1005, 1010, 1000, 1005, 1055, 1000, 1010, 1005, 2183, 1005, 1010, 1005, 2000, 1005, 1010, 1005, 2191, 1005, 1010, 1005, 1037, 1005, 1010, 1005, 17624, 1005, 1010, 1005, 2130, 1005, 1010, 1005, 3618, 1005, 1010, 1005, 2084, 1005, 1010, 1005, 7779, 1005, 1010, 1005, 29058, 8625, 13327, 1005, 1010, 1005, 1010, 1005, 1010, 1005, 3744, 1011, 18856, 19513, 1005, 1010, 1005, 3158, 1005, 1010, 1005, 5477, 4168, 1005, 1010, 1005, 2030, 1005, 1010, 1005, 7112, 1005, 1010, 1005, 16562, 2140, 1005, 1010, 1005, 1012, 1005, 1033]


The model needs to account for a few special tokens, namely the start + end of a sentence, unknown words and lastly for padding (each sentence has a different length, not well suited to feed into batches for a deep learning model so we set a suitable max length, then pad shorter sentences up to that length with a padding token.)  All this word is done for us using the `encode_plus` method, which we use to build our `Dataset` object.

```python
class SST_Dataset(Dataset):
    def __init__(self, ys, Xs, tokenizer, max_len):
        self.targets = ys
        self.reviews = Xs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }
```

Next we create our `Dataloader` objects for training, validation and testing. For each item in the dataset we need the encoded input tokens, masks for where the sentence is not padded and the target value.

```python
def create_data_loader(ys, Xs, tokenizer, max_len, batch_size):
    ds = SST_Dataset(ys, Xs, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size)

BATCH_SIZE = 16
MAX_LEN = 128

train_data_loader = create_data_loader(y_train, X_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(y_val, X_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(y_test, X_test, tokenizer, MAX_LEN, BATCH_SIZE)
```

## Constructing our model

Now we'ready to build our simple sentiment classification model: we use the output of the `DistilBertModel` - of size 768 - as input into a single fully-connected layer. Dropout is important here for a model with so many parameters (discussed below). (Hugging Face also provide some inbuilt models for downstream tasks that we could have used such as `BertForSequenceClassification` or `BertForQuestionAnswering`)

```python
class SentimentClassifier(nn.Module):
  def __init__(self, n_classes=5):
    super(SentimentClassifier, self).__init__()
    self.bert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids, attention_mask)
    output= output[0][:,0]
    output = self.drop(output)
    return self.fc(output)
```

The BERT authors had some recommendations for hyperparameters when it comes to fine-tuning:

*   *Batch size*: 16, 32
*   *Learning rate (Adam)*: 5e-5, 3e-5, 2e-5
*   *Number of epochs*: 2, 3, 4

We'll largely stick with these - note that the number of epochs is a lot lower than you might expect for a Deep Learning model. This is since we can easily overfit to the training set with many parameters. We'll check for this by calculating both the training and validation accuracy at each epoch. You can find out more about the Hugging Face's optimisers [here](https://huggingface.co/transformers/main_classes/optimizer_schedules.html).

```python
# initialise model
model = SentimentClassifier()

EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader)*EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=50,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)
```

Let’s continue with writing our helper functions for training our model. 

```python
def evalModel(model, data_loader, loss_fn, N):
    """Evaluate loss and accuracy of model on data_loader"""
    # set model to evaluation mode
    model = model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for d in data_loader:
            # get inputs and target 
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # pass through model + make prediction
            outputs = model(input_ids, attention_mask)
            _, pred = torch.max(outputs, dim=1)

            # update counters
            loss = loss_fn(outputs, targets)
            correct += (pred == targets).sum().item()
            total_loss += loss.item()*len(targets)

    # normalise
    return 100*correct/N, total_loss/N
```
```python
def trainModel(model, trainDataLoader, valDataLoader, loss_fn, optimizer, scheduler, verbose=True):
    """Train sentiment classifier"""
    # structure to store progress of the model at each epoch
    history = defaultdict(list)
    
    # move the model to the gpu
    model = model.to(device)

    for ep in range(EPOCHS):
        total_loss = 0
        correct = 0
        # set model to train mode so dropout and batch normalisation layers work as expected
        model.train()

        for d in trainDataLoader:
            # get inputs for batch
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # calculate output + loss
            model.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.squeeze(), targets.long())

            # take gradient step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # update losses
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == targets).sum().item()
            total_loss += loss.item()*len(targets)

        #after each epoch, collect statistics
        history['train_acc'].append(100*correct/len(X_train))
        history['train_loss'].append(total_loss/len(X_train))

        # statistics about the validation set
        val_acc, val_loss = evalModel(model, valDataLoader, loss_fn, len(X_val))
        history['val_acc'].append(val_acc)
        history['vall_loss'].append(val_loss)

        #if validation improved, save new best model
        if history['val_acc'][-1] == max(history['val_acc']):
            print ("=> Saving a new best at epoch:", ep)
            torch.save(model.state_dict(), 'best_model_state.bin')
        
        if verbose:
            print('Epoch {}/{}'.format(ep+1, EPOCHS))
            print('-' * 10)
            print('Train loss {} accuracy {}'.format(history['train_loss'][-1], history['train_acc'][-1]))
            print('Val loss {} accuracy {}'.format(val_loss, val_acc))

    #clean up
    model = model.to(torch.device("cpu"))
    del input_ids, attention_mask, targets, outputs, _, pred

    return model, history
```
Let's train our model and see how it does on our test set!

```python
%%time
best_model, histories = trainModel(model, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, verbose=True)
```
    => Saving a new best at epoch: 0
    Epoch 1/5
    ----------
    Train loss 1.1928647710812672 accuracy 47.588951310861425
    Val loss 1.3399161666658768 accuracy 41.416893732970024
    => Saving a new best at epoch: 1
    Epoch 2/5
    ----------
    Train loss 1.0168351954623553 accuracy 55.77013108614232
    Val loss 1.3073733697470267 accuracy 44.23251589464124
    Epoch 3/5
    ----------
    Train loss 0.8418751696633936 accuracy 64.68867041198502
    Val loss 1.4277723928563277 accuracy 43.23342415985468
    Epoch 4/5
    ----------
    Train loss 0.7150073092472687 accuracy 71.54728464419476
    Val loss 1.6733300179161883 accuracy 42.143505903723884
    Epoch 5/5
    ----------
    Train loss 0.6793975326396553 accuracy 73.67743445692884
    Val loss 1.7342694530045304 accuracy 41.87102633969119
    CPU times: user 10min 30s, sys: 5min 42s, total: 16min 13s
    Wall time: 16min 16s

```python
evalModel(best_model.to(device), test_data_loader, loss_fn, len(y_test))
```

    40.85972850678733 1.7384965674370123

There we have it! We've fine-tuned DistilBert for the task of sentiment classification to over 40% test accuracy in only 5 epochs. We can see that the pre-training step of this Tranformer model produces versatile, useful and high-quality features representing different semantics of language.

However we note that this doesn't get us close to [state-of-the-art](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained) on this dataset (55%) - the important lesson here is that we haven't tuned any hyperparameters so finding the best optimizer, learning-rate, droupout amount, adding hidden-layers + number of epochs is what will improve our model. We use the validation set to see what hyperparameters get the best accuracy on that - this estimates how our model will generalise to the unseen test set (see your favourite Learning Theory textbooka as to why this works).

Remember that during training we're trying to find the optima a (> 66,000,000 dimension) hypersurface - there's going to many minima so finding the best one requires some searching. Hyperparameter tuning is an important part of solving any problem with Machine Learning, one you just can't avoid.

![jpg](/images/bert_for_sst/hyperparam_meme.jpg)

As a final bit of fun, let's see what our model predicts on some raw text - we need to tokenise our custom input then pass it through our trained classifier. 
Though not a 5 we see the model can correctly identify the review as positive! 

```python
review_text = "I really loved Gladiator! It's my favourite film of all time. The cast was chosen perfectly."

encoded_review = tokenizer.encode_plus(
  review_text,
  max_length=MAX_LEN,
  add_special_tokens=True,
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',
  truncation=True
)

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)
output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
print(f'Review text: {review_text}')
print(f'Sentiment  : {int(prediction.cpu().detach().numpy())}')
```
    Review text: I really loved Gladiator! It's my favourite film of all time. The cast was chosen perfectly.
    Sentiment  : 4


Thanks for reading to the end of this post. As ever, for those interested, the Jupyter notebook with code for this post can be found on Github [here](https://github.com/kushmadlani/distilbert_for_sst). Here's some background reading & resources I found helpful:

## Further reading

*   [Visualising Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) post by Jay Allamar and its follow up [The Illustrated Tranformer](http://jalammar.github.io/illustrated-transformer/)
*   [State of transfer learning in NLP](https://ruder.io/state-of-transfer-learning-in-nlp/)
* [Lecture](https://www.youtube.com/watch?v=5vcj8kSwBCY) at Stanford, also found [this](https://youtu.be/S27pHKBEp30) video helpful
* The Hugging face Transformer library [docs](https://huggingface.co/transformers/)


