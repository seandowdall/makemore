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
print(words[: 10])
print(len(words))
print(min(len(w) for w in words))
print(max(len(w)for w in words))

# To begin we want to build a bigram langauge model. In this we will only be looking at 2 characters at a time
# we only look at one character that we are given and we are trying to predict the next character in a sequence
# e.g. what character is likely ot follow 'r'
# very simple but great way to start

for w in words[:3]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,  chs[1:]):
        print(ch1, ch2)