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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>policy</th>
      <th>candidate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>100% Clean Energy for America</td>
      <td>As published on Medium on September 3rd, 2019:...</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A Comprehensive Agenda to Boost America’s Smal...</td>
      <td>Small businesses are the heart of our economy....</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A Fair and Welcoming Immigration System</td>
      <td>As published on Medium on July 11th, 2019:\nIm...</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A Fair Workweek for America’s Part-Time Workers</td>
      <td>Working families all across the country are ge...</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A Great Public School Education for Every Student</td>
      <td>I attended public school growing up in Oklahom...</td>
      <td>warren</td>
    </tr>
  </tbody>
</table>
</div>



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
    data = text.lower()
    data = word_tokenize(data)
    data = [word for word in data if word not in stops]
    data = [stemmer.stem(token) for token in data]
    data = ' '.join(data)
    punctuation = string.punctuation + '—' + '’'
    data = data.translate(str.maketrans('', '', punctuation))
    data = data.replace('  ', ' ')
    return data.strip()
```

**For this warmup, tests are not provided.** 

Instead, examine the output for the following cell. 
- Was your code successful? 
- Are there words in the output that should be added to our list of stopwords?
- Should we remove numbers?


```python
preprocessing(df.policy[0])
```




    'publish medium septemb 3rd 2019 scienc clear world lead expert long known climat chang caus human be  acceler alreadi see effect record flood devast wildfir 100year storm happen everi year cost live caus hundr billion dollar damag disproportion impact vulner commun includ commun color children lowincom commun emerg without rapid widespread chang soon unabl prevent worst harm climat crisi leav one untouch also repres onceinagener opportun creat million good american job clean renew energi infrastructur manufactur unleash best american innov creativ rebuild union creat real progress justic worker directli confront racial econom inequ embed fossil fuel economi world must limit warm 15° c avoid catastroph outcom cut carbon pollut roughli half 2030 achiev netzero emiss 2050 world largest histor carbon pollut unit state special respons lead way  origin support green new deal challeng us go beyond launch tenyear mobil 2030 achiev netzero domest greenhous ga emiss fast possibl  also woven climat mitig resili throughout polici propos  meet climat goal onetim onesizefitsal approach  need big structur chang across rang industri sector sustain time idea begin must take bold action confront climat crisi import urgent anyth els next presid face wait add name agre need big structur chang across rang industri sector solv climat crisi email zip submit presidenti candid friend governor jay insle challeng american confront urgenc climat crisi bear upon us jay mere sound alarm make vagu promis provid bold thought detail idea get us need go rais standard address pollut invest futur american economi presidenti campaign may idea remain center agenda one import idea urgent need decarbon key sector economi today  embrac goal commit adopt build governor insle tenyear action plan achiev 100 clean energi america decarbon electr vehicl build  challeng everi candid presid governor insle note 100 clean energi america plan tackl key sector economi make immedi differ electr transport build relat commerci activ respons nearli 70 percent us carbon emiss technolog today start construct cleaner grid modern auto industri green commun green apollo plan invest 400 billion ten year clean energi r spur innov help us develop technolog need go final mile critic condit r invest result manufactur take place right america creat good middleclass job green manufactur plan invest 15 trillion ten year feder procur americanmad clean energi product fund transit feder state local govern plan public land make unpreced commit gener 10 overal electr need renew sourc offshor public land green marshal plan commit 100 billion support export americanmad clean energi product help countri cut emiss today  announc commit addit 1 trillion 10 year fulli paid revers trump tax cut wealthiest individu giant corpor match governor insle commit subsid econom transit clean renew electr zero emiss vehicl green product commerci residenti build told feder invest 3 trillion leverag addit trillion privat invest creat million job achiev 2028 100 zerocarbon pollut new commerci residenti build 2030 100 zero emiss new lightduti passeng vehicl mediumduti truck buse 2035 100 renew zeroemiss energi electr gener interim target 100 carbonneutr power 2030 goal may decad away work achiev must start congress act immedi reduc emiss begin transit clean renew power  also lot presid  take decis action first term use power clean air act author set high regulatori standard impos ambiti interim target along way noth less nation mobil requir defeat climat chang requir everi singl one us  time roll sleev get work time wast help campaign keep fight re count grassroot donor make campaign possibl ve save inform actblu express donat go immedi  15 28 50 100 250 partner worker ensur one left behind task us huge demand us requir retrofit nation build reengin electr grid adapt manufactur base increas product zero emiss vehicl also requir readjust econom approach ensur commun color other systemat exclud fossil fuel economi left behind transit clean energi also opportun  need million worker peopl know build thing manufactur skill experienc contractor plan execut larg construct engin project train joint labor manag apprenticeship ensur continu suppli skill avail worker succeed fight climat chang unless peopl skill get job done room full partner  ensur benefit uplift empow worker may hurt transit green economi includ coal worker other current employ fossil fuel industri mean provid financi secur includ earli retir benefit job train union protect benefit guarante wage benefit pariti affect worker moreov longer forc worker make imposs choic fossil fuel industri job superior wage benefit green economi job pay far less job creat union job accompani pay scale benefit long tension transit green economi creat good middl class union job warren administr thing look away frontlin commun endur air water pollut decad disproportion threaten acceler climat chang everyon right breath clean air drink clean water live healthi commun free pollut today commun color lowincom commun disproportion impact environment hazard impact even greater children afford perpetu structur inequ embed exist system move toward clean energi futur respons must priorit resourc peopl commun left behind fossil fuel economi hit worst climat chang pollut  ensur invest spur econom develop everi part countri coast  invest communitydevelop project peopl live work area best know need  creat truli participatori democrat process center led live front line climat chang promis green new deal lift frontlin commun empow worker tackl climat chang time  decarbon sector contribut pollut problem  work 100 clean electr electr consumpt contribut one third carbon pollut today good news renew fastest grow sourc electr gener renew energi continu drop cost mani part world alreadi cheaper fossil fuel even without subsidi state includ home state massachusett governor insle home state washington alreadi lead way adopt technic challeng around transmiss storag remain increas renew energi invest smart grid advanc distribut improv reliabl bring cost open new econom opportun achiev 100 clean electr also help us decarbon transport build put place project labor agreement protect ensur job creat union job accompani pay scale benefit presid  work rapidli achiev 100 clean renew zeroemiss energi electr gener  set high standard util nationwid administr requir util achiev 100 carbonneutr power 2030 strong interim target along way achiev allclean renew zeroemiss energi electr gener 2035 also establish regul retir coal power within decad ensur leav coal commun behind fund health care pension miner creat feder renew energi commiss  work congress overhaul feder energi regulatori commiss task regul us electr grid replac instead feder renew energi commiss revis commiss mission reduc greenhous ga pollut  slam shut revolv door industri ensur respons fossil fuel interest commun use strength feder invest polici acceler transit  requir feder agenc achiev 100 clean energi domest power purchas end first term commit public land plan  set goal provid 10 overal electr gener renew sourc offshor public land nearli ten time current gener provid feder subsidi speed clean energi adopt  expand exist feder energi financ program like depart energi loan guarante program rural util servic includ provid direct grant clean energi project  extend program provid grant lieu tax credit establish refund tax incent speed util deploy exist smart grid advanc transmiss technolog work util increas onbil invest energi effici solut includ subsid invest lowincom commun  implement commun workforc projectlabor agreement ensur job creat invest good union job prevail wage determin collect bargain expand interst region coordin maxim effici grid  provid incent expedit plan site longdist interst transmiss clean electr  priorit area signific queue cleanenergi gener capac await transmiss  provid dedic support four power market administr tennesse valley author appalachian region commiss help build publiclyown clean energi asset deploy clean power help commun transit fossil fuel  expand invest smart energi storag solut cybersecur grid 100 clean vehicl market electr vehicl expand rapidli around world rang increas cost electr car batteri drop precipit averag zeroemiss vehicl alreadi cheaper drive car run gasolin instead acceler innov sector trump administr busi attempt loosen automobil emiss standard even us auto industri protest meanwhil european union tighten standard china alreadi world largest produc electr vehicl make aggress inroad domin global market least 10 countri set aggress electr vehicl target elimin carbonemit car entir remain competit help save planet american automobil manufactur must keep warren administr set goal achiev zero emiss new lightduti passeng vehicl mediumduti truck buse 2030 achiev  set ambiti standard fuel emiss first year offic  set strict vehicl emiss standard becom progress tighter everi year reach requir 100 zeroemiss new light mediumduti vehicl 2030 time  establish clean fuel standard reduc greenhous ga emiss promot lowercarbon altern fuel modern automot manufactur base develop need infrastructur  provid feder invest grow domest zeroemiss vehicl manufactur reinforc assembl plant suppli base includ batteri manufactur  also invest electr vehicl charg infrastructur includ ensur everi feder interst highway rest stop host fastcharg station end first term offic ensur charg station widespread access tomorrow ga station today boost consum demand zero emiss vehicl  extend busi consum tax credit purchas zeroemiss vehicl  creat “ clean car clunker ” program base recoveri act tradein program extend financi incent encourag consum replac fuelineffici car zeroemiss vehicl made america union worker green manufactur plan commit 15 trillion ten year feder procur clean green americanmad product includ zeroemiss vehicl  use fund requir rapid electrif feder vehicl fleet requir new vehicl purchas zeroemiss end first term  work state local govern acceler electrif vehicl fleet well includ financ transit diesel zeroemiss transit school buse decarbon form transit stop car buse must address carbon pollut form transport includ maritim rail aviat expand improv public transit across countri addit transform vehicl sector administr invest research priorit decarbon longdist ship transport two challeng sector decarbon aviat pollut particular remain fastgrow presid  commit intern goal hold climat pollut civil aviat 2020 level reduc time 100 clean build build construct account onethird global energi use carbon emiss embodi carbon includ world continu urban emiss grow unit state spend 400 billion year provid electr heat home busi make signific progress decarbon build achiev clean electr gener reduc energi wast increas energi effici improv air qualiti health outcom creat good union construct job job outsourc sent oversea presid  commit take immedi action achiev zerocarbon pollut new commerci residenti build end second term 2028 make happen  adopt bold standard construct  creat nation zerocarbon build standard 2023  partner state local govern enforc new stronger build code administr provid incent local govern adopt aggress standard bring emiss  link energi pollut standard feder support new construct project build agenc grantmak requir feder hous tax credit green mortgag product offer feder hous financ agenc  direct feder agenc acceler proven applianc energi effici standard make americanmanufactur applianc cleaner competit save consum money use feder buy power drive chang  use power feder govern shift market acceler adopt rule elimin fossil fuel use new renov feder build move deadlin five year end first term 2025 use portion 15 trillion feder procur commit green manufactur plan purchas clean energi product use feder build construct materi heat storag technolog applianc  increas access feder financ retrofit new construct upgrad public build level govern encourag privat capit invest  creat incent privat invest energi effici electrif residenti commerci build includ tax credit direct spend regulatori tool  expand refund credit instal energi effici upgrad extend exist tax credit wind solar power  make easier institut capit invest portfolioscal green construct retrofit scale clean energi larg commerci residenti project incentiv retrofit exist build stock addit achiev zero emiss new build must address exist stock commerci build residenti hous green hous perk avail wealthi must made afford everyon  establish nation initi upgrad build energi effici offer tax credit gener inclus financ direct feder fund put american work reduc carbon output exist home busi includ subsid weather lowincom household  meet governor insle target refurbish 4 hous build everi year job done work togeth make smart invest clean energi futur grow economi improv health reduc structur inequ embed exist fossil fuel system task us monument urgent  confid america tool knowhow make happen presid take bold action confront climat crisi start day one futur planet depend time wast'



**Let's apply our preprocessing to every policy.**


```python
df.policy = df.policy.apply(preprocessing)

print(df.policy[:3])
```

    0    publish medium septemb 3rd 2019 scienc clear w...
    1    small busi heart economi even though american ...
    2    publish medium juli 11th 2019 immigr alway vit...
    Name: policy, dtype: object


Now, we can explore our text data.

In the cell below define a function called `average_word_length` that receives a single parameter called `text`, and outputs the average word length.

<u><b>This function should:</b></u>
1. Split the text into a list of tokens
2. Find the length of every word in the list
3. Sum the word lengths and divide by the number of words in the list of tokens.
4. Return the result.


```python
def average_word_length(text):
    split = text.split()
    word_lengths = [len(x) for x in split]
    average = sum(word_lengths)/len(split)
    return average
```

Now, we apply our function to every policy and add the output as column.


```python
df['average_word_length'] = df.policy.apply(average_word_length)
```

Sweet let's take a look at the documents with the highest average word length.


```python
df.sort_values(by='average_word_length', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>policy</th>
      <th>candidate</th>
      <th>average_word_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69</th>
      <td>69</td>
      <td>Tackling the Climate Crisis Head On</td>
      <td>climat plan clean air water clean energi corpo...</td>
      <td>warren</td>
      <td>6.137931</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>A Working Agenda for Black America</td>
      <td>tabl content address matern mortal afford hous...</td>
      <td>warren</td>
      <td>6.040816</td>
    </tr>
    <tr>
      <th>62</th>
      <td>62</td>
      <td>Restoring America’s Promise to Latinos</td>
      <td>tabl content afford hous bankruptci child care...</td>
      <td>warren</td>
      <td>5.960000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>Fighting Digital Disinformation</td>
      <td>sinc 2016 elect investig congression hear acad...</td>
      <td>warren</td>
      <td>5.885246</td>
    </tr>
    <tr>
      <th>95</th>
      <td>95</td>
      <td>Responsible Foreign Policy</td>
      <td>key point detail us must lead world improv int...</td>
      <td>sanders</td>
      <td>5.876033</td>
    </tr>
  </tbody>
</table>
</div>



An average measurement can be a bit misleading. 

Let's also write a function that finds the word count for a given document.

In the cell below, define a function called `word_count` that receives a single `text` parameter.

<u><b>This function should:</b></u>
1. Split the text data
2. Return the length of the array.


```python
def word_count(text):
    split = text.split()
    return len(split)
```

Nice. Now we apply the function to our entire dataset, and save the output as a column


```python
df['word_count'] = df.policy.apply(word_count)

df.sort_values(by='average_word_length', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>policy</th>
      <th>candidate</th>
      <th>average_word_length</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69</th>
      <td>69</td>
      <td>Tackling the Climate Crisis Head On</td>
      <td>climat plan clean air water clean energi corpo...</td>
      <td>warren</td>
      <td>6.137931</td>
      <td>29</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>A Working Agenda for Black America</td>
      <td>tabl content address matern mortal afford hous...</td>
      <td>warren</td>
      <td>6.040816</td>
      <td>49</td>
    </tr>
    <tr>
      <th>62</th>
      <td>62</td>
      <td>Restoring America’s Promise to Latinos</td>
      <td>tabl content afford hous bankruptci child care...</td>
      <td>warren</td>
      <td>5.960000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>Fighting Digital Disinformation</td>
      <td>sinc 2016 elect investig congression hear acad...</td>
      <td>warren</td>
      <td>5.885246</td>
      <td>1281</td>
    </tr>
    <tr>
      <th>95</th>
      <td>95</td>
      <td>Responsible Foreign Policy</td>
      <td>key point detail us must lead world improv int...</td>
      <td>sanders</td>
      <td>5.876033</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>



Interesting. Let's take a look at the distribution for the `word_count` column.


```python
warren_df = df[df.candidate=='warren']
sanders_df = df[df.candidate=='sanders']

plt.hist(warren_df.word_count, alpha=.6, label='Warren')
plt.hist(sanders_df.word_count, alpha=.6, label='Sanders')
plt.legend()
plt.show()
```


![png](index_files/index_22_0.png)


It looks like the average length of a policy is about 1,000 words.

Let's print the mean and median for the `word_count` column.


```python
print('Mean Word Count: ',df.word_count.mean())
print('Median Word Count: ',df.word_count.median())
```

    Mean Word Count:  1625.697247706422
    Median Word Count:  1098.0


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

    [('feder', 865), ('plan', 716), ('make', 706), ('peopl', 704), ('american', 701), ('health', 653), ('govern', 636), ('also', 630), ('care', 629), ('commun', 611)]


Are there any words here that should be added to our list of stopwords?

*In the cell below* define a function called `word_frequency` that receives a series of documents, and outputs a fitted FreqDist object.

<u><b>This function should be</b></u> a generalized version of the code we just wrote, only instead of printing out the most frequent words, the function should return the `fd` object.


```python
def word_frequency(documents):
    tokens = [word_tokenize(document) for document in documents]
    bow = []
    for doc in tokens:
        bow.extend([word.lower() for word in doc])
    
    fd = FreqDist(bow)
    return fd
```

Now, we can feed all of sanders policies into our `word_frequency` functions and receive a fitted `FreqDist` object


```python
fd_sanders = word_frequency(sanders_df.policy)
fd_sanders.most_common(10)
```




    [('peopl', 433),
     ('ensur', 410),
     ('disabl', 400),
     ('berni', 357),
     ('commun', 354),
     ('provid', 343),
     ('care', 338),
     ('fund', 331),
     ('program', 312),
     ('worker', 292)]



`FreqDist` objects come with a handy `.plot` method :)


```python
fd_sanders.plot(10);
```


![png](index_files/index_38_0.png)

