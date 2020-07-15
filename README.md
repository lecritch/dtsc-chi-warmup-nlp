# Text Exploration
![](https://media.giphy.com/media/3o6ozjrPeWQifzyA6Y/giphy.gif)

This morning we will be doing some EDA with the nltk library.


```python
# Base Libraries
import pandas as pd
import numpy as np
import string

# NLP
import nltk
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.collocations import (BigramCollocationFinder, BigramAssocMeasures, 
                               TrigramCollocationFinder, TrigramAssocMeasures)

# Visualization
import matplotlib.pyplot as plt
```

In the cell below, we import the policy proposal by 2020 Democratic Presidential Candidates Bernie Sanders and Elizabeth Warren.


```python
df = pd.read_csv('data/2020_policies_feb_24.csv')
df.head()
```

**We need to do some processing to make this text usable.** 

In the cell below, define a function called `prepocessing` that receives a single parameter called `text`.

<u><b>This function should:</b></u>
1. Lower the text so all letters are the same case
2. Use nltk's `word_tokenize` function to convert the string into a list of tokens.
3. Remove stop words from the data using nltk's english stopwords corpus.
4. Use nltk's `PortStemmer` to stem the text data
5. Remove punctuation from the data 
    - *(You can use the [string](https://www.journaldev.com/23788/python-string-module) library for this)*
6. Convert the list of tokens into a string
7. Remove opening and trailing spaces, and replace all double spaces with a single space.
8. Return the results.


```python
stemmer = PorterStemmer()
stops = stopwords.words('english')

def preprocessing(text):
    pass
```

**For this warmup, tests are not provided.** 

Instead, examine the output for the following cell. 
- Was your code successful? 
- Are there words in the output that should be added to our list of stopwords?
- Should we remove numbers?


```python
preprocessing(df.policy[0])
```

**Let's apply our preprocessing to every policy.**


```python
df.policy = df.policy.apply(preprocessing)

print(df.policy[:3])
```

Now, we can explore our text data.

In the cell below define a function called `average_word_length` that receives a single parameter called `text`, and outputs the average word length.

<u><b>This function should:</b></u>
1. Split the text into a list of tokens
2. Find the length of every word in the list
3. Sum the word lengths and divide by the number of words in the list of tokens.
4. Return the result.


```python
# Your code here

```

Now, we apply our function to every policy and add the output as column.


```python
df['average_word_length'] = df.policy.apply(average_word_length)
```

Sweet let's take a look at the documents with the highest average word length.


```python
df.sort_values(by='average_word_length', ascending=False).head()
```

An average measurement can be a bit misleading. 

Let's also write a function that finds the word count for a given document.

In the cell below, define a function called `word_count` that receives a single `text` parameter.

<u><b>This function should:</b></u>
1. Split the text data
2. Return the length of the array.


```python
# Your code here

```

Nice. Now we apply the function to our entire dataset, and save the output as a column


```python
df['word_count'] = df.policy.apply(word_count)

df.sort_values(by='average_word_length', ascending=False).head()
```

Interesting. Let's take a look at the distribution for the `word_count` column.


```python
warren_df = df[df.candidate=='warren']
sanders_df = df[df.candidate=='sanders']

plt.hist(warren_df.word_count, alpha=.6, label='Warren')
plt.hist(sanders_df.word_count, alpha=.6, label='Sanders')
plt.legend()
plt.show()
```

It looks like the average length of a policy is about 1,000 words.

Let's print the mean and median for the `word_count` column.


```python
print('Mean Word Count: ',df.word_count.mean())
print('Median Word Count: ',df.word_count.median())
```

*There are some outliers in this data in a full data science project would need to be explored.*

**Ok, final thing!**

Let's find out what the most frequent words are for each candidate.

First, we use list comprehension to create a list of token-lists.


```python
token_warren= [word_tokenize(policy) for policy in warren_df.policy] 
```

Next, we want to create a bag of words. AKA a single list containing every token.


```python
warren_bow = []
for doc in token_warren:
    warren_bow.extend([word.lower() for word in doc])
```

Now, we use the `FreqDist` object to find the 10 most frequent words.


```python
fd_warren = FreqDist(warren_bow)
print(fd_warren.most_common(10))
```

Are there any words here that should be added to our list of stopwords?

*In the cell below* define a function called `word_frequency` that receives a series of documents, and outputs a fitted FreqDist object.

<u><b>This function should be</b></u> a generalized version of the code we just wrote, only instead of printing out the most frequent words, the function should return the `fd` object.


```python
# Your code here

```

Now, we can feed all of sanders policies into our `word_frequency` functions and receive a fitted `FreqDist` object


```python
fd_sanders = word_frequency(sanders_df.policy)
fd_sanders.most_common(10)
```

`FreqDist` objects come with a handy `.plot` method :)


```python
fd_sanders.plot(10);
```


```python

```
