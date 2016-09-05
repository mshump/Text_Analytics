#################################################
## Text Analysis | Data Science                ##
## Some Basic packages, techniques and outputs ##
#################################################

##X# install packages if you do not already have

#install.packages("tm")
#install.packages("openNLPmodels.en", repos = "http://datacube.wu.ac.at/", type = "source")

##X# load packages for this session

#library(tm)

libs <- c("RODBC","RWeka", "wordcloud","tm", "topicmodels", "plyr", 
                "stringr","openNLPmodels.en","openNLP", "NLP")
lapply(libs, require, character.only=TRUE)

##X# set directory for this session

setwd("C:/Users/mshump/Desktop/Text Data/")  ## notice / NOT \

##X# read the data
data <- read.csv("SMSSpamCollection.csv")

# check out your data
str(data)
head(data)
tail(data)

# convert factor to text
data$text <- as.character(data$text)
str(data)


##X#  construct a Corpus
corpus <- Corpus(VectorSource(data$text))

inspect(corpus)

##X#  clean - up

cleanCorpus <- function(corpus){
        corpus.tmp <- corpus
        corpus.tmp <- tm_map(corpus, removePunctuation)
        corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
        corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower))
        corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("English"))
        return(corpus.tmp)
        
}

corpus.cl <- cleanCorpus(corpus)

##X#  TDM - term document matrix
tdm <- TermDocumentMatrix(corpus.cl)
inspect(tdm)

##X# build the document term matrix:
dtm <- DocumentTermMatrix(corpus.cl)
inspect(dtm)

rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm.new   <- dtm[rowTotals> 0, ] #remove all docs without words


##X#  Below we specify that we want terms / words which were used 8 or more times (in all documents / paragraphs).

findFreqTerms(x = dtm.new, lowfreq = 100, highfreq = Inf)

##X# Finding words which 'associate' together.

findAssocs(x = dtm.new, term = "free", corlimit = 0.25)

##X#  If desired, terms which occur very infrequently (i.e. sparse terms) can be removed

dtm.common.99 <- removeSparseTerms(x = dtm.new, sparse = 0.99)
dtm.common.95 <- removeSparseTerms(x = dtm.new, sparse = 0.95)

dtm.new                 # 9249 terms
dtm.common.99           # 119 terms
dtm.common.95           # 6 terms

inspect(dtm.common.99[1:5,])
inspect(dtm.common.95[1:5,])


###############


##X#  start with TDM
tdm
class(tdm)

##X# remove extreme sparse and create matrix
tdm.common.99 <- removeSparseTerms(x = tdm, sparse = 0.99)
tdm_cl <- as.matrix(tdm.common.99)

##X# get word counts in decreasing order and then a list of the words

word_freqs <- sort(rowSums(tdm_cl), decreasing=TRUE)
        length(word_freqs)
smalldict <- PlainTextDocument(names(word_freqs))
        smalldict

##X#  ngrams 

data("crude")
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm <- TermDocumentMatrix(crude, control = list(tokenize = BigramTokenizer))
inspect(tdm[340:345,1:10])

inspect(tdm)
inspect(crude)


##X#  Sentiment

pos.words <- c('good', 'great')
neg.words <- c('bad', 'terrible') 

sample <- c("You're good and bad", "You are terrible", "you great")

score.sentiment = function(sentences, pos.words, neg.words, .progress= 'none')
        
{        require(plyr)        
         require(stringr)	
         # we got a vector of sentences. plyr will handle a list or a vector as an "l" for us	
         # we want a simple array of scores back, so we use "l" + "a" + "ply" = laply:	
         scores = laply(sentences, function(sentence, pos.words, neg.words)  {	
                 
                 # clean up sentences with Rs regex-driven global substitute, gsub():	
                 sentence = gsub('[[:punct:]]', '', sentence)	
                 sentence = gsub('[[:cntrl:]]', '', sentence)	
                 #sentence = gsub('\\d+', , sentence)	
                 # and convert to lower case:	
                 sentence = tolower(sentence)
                 
                 # split into words. str_split is in the stringr package
                 word.list = str_split(sentence, '\\s+')	
                 # sometimes a list() is one level of hierarchy too much	
                 words = unlist(word.list)
                 
                 # compare our words to the dictionaries of positive & negative terms	
                 pos.matches = match(words, pos.words)	
                 neg.matches = match(words, neg.words)	
                 
                 # match() returns the position of the matched term or NA	
                 # we just want a TRUE/FALSE:	
                 pos.matches = !is.na(pos.matches)	
                 neg.matches = !is.na(neg.matches)
                 
                 # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():	
                 score = sum(pos.matches) - sum(neg.matches)	
                 
                 return(score)	
         }, pos.words, neg.words, .progress=.progress )	
         
         scores.df = data.frame(score=scores, text=sentences)	
         return(scores.df)
}


result <- score.sentiment(sample, pos.words,  neg.words)
class(result)
result$score



##X#  Tagging POS and Entity

## Requires package 'openNLPmodels.en' from the repository at
## <http://datacube.wu.ac.at>.
#install.packages("openNLPmodels.en", repos = "http://datacube.wu.ac.at/", type = "source")
#require("openNLPmodels.en")

##X# POS tag
s <- paste(c("Pierre Vinken, 61 years old, will join the board as a ",
             "nonexecutive director Nov. 29.\n",
             "Mr. Vinken is chairman of Elsevier N.V., ",
             "the Dutch publishing group."),
           collapse = "")
s <- as.String(s)
##X#  Chunking needs word token annotations with POS tags.
sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
pos_tag_annotator <- Maxent_POS_Tag_Annotator()
a3 <- annotate(s,
               list(sent_token_annotator,
                    word_token_annotator,
                    pos_tag_annotator))
annotate(s, Maxent_Chunk_Annotator(), a3)
annotate(s, Maxent_Chunk_Annotator(probs = TRUE), a3)

##X#  Entity 

## Need sentence and word token annotations.
sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
## Entity recognition for persons.
entity_annotator <- Maxent_Entity_Annotator()
entity_annotator
annotate(s, entity_annotator, a2)
## Directly:
entity_annotator(s, a2)
## And slice ...
s[entity_annotator(s, a2)]
## Variant with sentence probabilities as features.
annotate(s, Maxent_Entity_Annotator(probs = TRUE), a2)



## Need sentence and word token annotations.
sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
pos_tag_annotator <- Maxent_POS_Tag_Annotator()
pos_tag_annotator

a4 <- annotate(s, list(sent_token_annotator, word_token_annotator,pos_tag_annotator))

## Determine the distribution of POS tags for word tokens.
a3w <- subset(a3, type == "word")
tags <- sapply(a3w$features, `[[`, "POS")
tags
table(tags)
## Extract token/POS pairs (all of them): easy.
sprintf("%s/%s", s[a3w], tags)



