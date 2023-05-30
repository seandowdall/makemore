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

# load up dataset
words = open('names.txt', 'r').read().splitlines()
# print(words[: 10])
# print(len(words))
# print(min(len(w) for w in words))
# print(max(len(w)for w in words))

# To begin we want to build a bigram langauge model. In this we will only be looking at 2 characters at a time
# we only look at one character that we are given, and we are trying to predict the next character in a sequence
# e.g. what character is likely ot follow 'r'
# very simple but great way to start

b = {}  # dictionary 'b' to maintain counts for every one of the bigrams
# each 'w' here is an individual word (a string)
for w in words:
    chs = ['<S>'] + list(w) + [
        '<E>']  # created a special array here (characters), hallucinated a special start token + 'W' (string 'emma') + end special token
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1  # add up all bigrams and count hwo often they will occur
        # print(ch1, ch2)
print(sorted(b.items(), key=lambda kv: -kv[
    1]))  # returns the tuples that occurred most frequently in descending order. E.g. ('n', '<E>'), 6763 (n was the last letter 6736 times)
# individual counts that we achieve over the entire dataset
