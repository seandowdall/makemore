import torch
import matplotlib.pyplot as plt

# Make more, makes more of things that you give it. We will be working with a large dataset of names and building a language model which can make unique names.
# Under the hood, make-more is a character level language model. It is treating every line like an example and within each example its treating them all as sequences of individual characters.
# NEURAL NETS WE WILL IMPLEMENT:
# Bigram
# Bag Of Words
# MLP (Multi-Layer Perceptron)
# Recurring Neural Networks (RNN)
# GRU
# Modern Transformers (equivalent to GPT-2)

# USEFUL CODE FOR DOWNLOADING AND OUTPUTTING FILES FROM GITHUB USING REQUESTS
# ---------------------------------------------------------------------------
# import requests
# url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
# filename = "names.txt"
#
# response = requests.get(url)
# response.raise_for_status()  # Check for any errors
#
# with open(filename, "w") as file:
#     file.write(response.text)
#
# print(f"File '{filename}' downloaded and saved successfully.")
# ------------------------------------------------------------------


# load up dataset
words = open('names.txt', 'r').read().splitlines()
# print(words[: 10])
# print(len(words)) OUTPUT 32033
# print(min(len(w) for w in words)) Shortest word: 2
# print(max(len(w)for w in words)) longest word: 15 characters

# What does the Word: ISABELLA tell us?
#  1. 'I' is very likely to be the first letter of the name
#  2. 'S' is very likley to follow 'I'
#  3. 'A' is very likley to follow 'IS'
#  4. 'B' is very likely to follow 'ISA'
#  5. 'E' is very likley to follow 'ISAB'
#  6. 'L' likely to follow 'ISABE'
#  7. 'L' likely to follow 'ISABEL'
#  8. 'A' likely to follow 'ISABELL'
#  9. AFTER 'ISABELLA' : Word is very likley to end

# instead of doing this only for a single word... We now have to model 32000 words

# To begin we want to build a bigram langauge model. In this we will only be looking at 2 characters at a time
# we only look at one character that we are given, and we are trying to predict the next character in a sequence
# e.g. what character is likely to follow 'r'
# always just look at previous character to predict the next one
# very simple but great way to start

b = {}  # dictionary 'b' to maintain counts for every one of the bigrams
# each 'w' here is an individual word (a string)
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
# created a special array here (characters), hallucinated a special start token + 'W' (string 'emma') + end special token
    for ch1, ch2 in zip(chs, chs[1:]):  # iterate through the words
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1  # add up all bigrams and count how often they will occur
        # print(ch1, ch2)
# print(b) this would show us the count of individual characters across the entire dataset
# print(sorted(b.items(), key=lambda kv: -kv[1]))  # usually sort will sort on first item of a touple, but we want to perfomr on the key value
# returns the tuples that occurred most frequently in descending order. E.g. ('n', '<E>'), 6763 (n was the last letter 6736 times)
# individual counts that we achieve over the entire dataset


# it will actually be significantly more convenient for us to keep this information in a 2d array instead of a python dictionary
# we are going to store this information in a 2d array and the rows are going to be the 1st character and the columns will be the second character
# each entry in this 2d array will tell us how often that 2nd char follows the 1st char in the dataset
# to do this we will use pytorch - deep learning nn framework and torch.tensor which allows us to create multidimensional arrays and manipulate them very efficiently

# a = torch.zeros((3,5))
# print(a)
# OUTPUT: tensor([[0., 0., 0., 0.],
#                 [0., 0., 0., 0.],
#                 [0., 0., 0., 0.]])
# print(a.dtype) OUTPUT: torch.float32 - Gives us our array of zeros in floating point value. We want this in int

# so here we use dtype to convert to 32-bit integers
# tensors allow us to manipulate all the individual entries very efficiently

N = torch.zeros((27, 27), dtype=torch.int32)  # 27 * 27 array of zeros, N represents counts here
chars = sorted(list(set(''.join(words))))  # ''.join(words) --> concatenate all words and makes them one massive string - pass to set constructor, throws out all duplicate characters - list sorted from a-z
stoi = {s: i + 1 for i, s in enumerate(chars)}  # now a starts at 1 'a': 1,'b': 2 ........'z':26, '.':0
stoi['.'] = 0  # only one special token now - has position 0
# first thing we need to do before visualising the data is invert the array above
itos = {i: s for s, i in stoi.items()}  # don't need to change this as it is just a reverse mapping

# here we replace our dictionary 'd' with our new array 'N'
for w in words:
    chs = ['.'] + list(w) + [
        '.']  # created a special array here (characters), hallucinated a special start token + 'W' (string 'emma') + end special token
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# for visualizing in a jupyter notebook
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

# row of counts of all the first letters (x-axis)
# Y axis is all the letters that end a sentence
# now we follow the counts and start sampling for the model

# print(N[0,:])  # indexes into the zeroth row, and colon grabs all the columns  # OUTPUT; tensor([   0, 4410, 1306, 1542, 1690, 1531,  417,  669,  874,  591, 2422, 2963,
# 1572, 2538, 1146,  394,  515,   92, 1639, 2055, 1308,   78,  376,  307,
#  134,  535,  929], dtype=torch.int32)

# THESE are the counts, and now we would like to sample from this: so we convert this to probabilities
# so, we create a probability vector
p = N[0].float()  # int converted to float so that we can normalise these counts
p = p / p.sum()  # this creates a vector of smaller numbers (now we have probabilities) - p.sum() is now 1 - its is a proper probability distribution. Gives us the probability for any giuven character to be the first character of a word
# print(p)  # OUTPUT: tensor([0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273,
# 0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029,
# 0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290])

# now we can try to sample from these distributions
# we will use torch.multinomial - returns samples from the multinomial probability distributions
# in other words: you give me probabilities, I give you integers which are sampled according to the probability distribution
#
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# print(itos[ix])

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
# print(p)
# output: tensor([0.6064, 0.3033, 0.0903])
# this mean that 0.6% of results should be zero, 0.3% should be 1, 0.09% should be 2s
# print(torch.multinomial(p, num_samples=100, replacement=True, generator=g))
# use torch.multinomail to draw samples from it - ask for num samples, replacement true means that when we draw an element then we can put it back in an eligible list of indices to draw again
# output: tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0,
#         1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
#         0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
#         0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0,
#         0, 1, 1, 1])
# lots of zeros, 2 times as few ones and 3 times as few twos

# grab floating point value of N
P = N.float()
# then we want to divide all rows so that they sum to 1
P = P / P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)
# we are going to have to get very good at tensor manipulations moving forward
for i in range(50):

    out = []
    ix = 0
    while True:
        p = P[ix]
        # p = N[ix].float()
        # p = p / p.sum()
        # p = torch.ones(27)/27
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append((itos[ix]))
        if ix == 0:
            break
    print(''.join(out))
