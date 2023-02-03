# Guessing the party of the Senator who wrote a tweet
Comparing different machine learning models 
 
## Part 0: Major Revisions Summary

My first main goal for my revisions to my work on this task was to streamline my files, removing artifacts, half completed tests, extraneous variables, and other stuff that was making it confusing to read through. I also restructured parts of my code and added more explanation into my jupyter notebook, so what I was doing made more sense. 

I also added a lot of parameter refining steps and functions. In my code, I played around a bit with optimizing parameters but didn't do a great job recording my work and progress. In my revised version, I added several "tester" functions to my python script that mostly use accuracy\_scores to measure parameter success and create graphs so that I can easily see which parameters are leading to good results. Doing my parameter refinements mroe concisely helped me explain and stremaline my code. 

## Part 1: Goal
The goal of this code is to compare two possible machine learning models to find the best one to categorize tweets from US Senators into what political party that Senator is in (Democrat or Republican) based on its qualities.

My code implements a variety of machine learning algorithms in python and assesses their efficacy. It also compares and assesses the efficacy of machine learning algorithms and results in terms of the context of the data’s domain The main goal of my code is to try and identify and create the two best possible models to solve the classification problem presented by my tweet data, and then to assess those models efficacy first to optimize their performances separately and then to pick which of the two is the most successful at the task.  

## Part 2: Making my Tree

I picked a decision tree for this task because it seemed intuitive to me to break text data up by the columns and then go through them sequentially in a branching way to categorize them. I chose to used binary splits because I realized I got a lot more out of the complex categories in my dataset (user, number of favorites, state..) into one binary characteristic I was interested in pulling from them. For state, I broke it up based on how they voted in the last election. For senator, I broke it up based on gender. This was also helpful because it made me narrow down what I was most interested in pulling from my spreadsheet, which was massive and had lots of angles from which I could approach it. In this code, I streamlined the function that was converting my columns to binary because I realized I didn't want to save every iteration of my dataset as it's own variable. I also added an additional evaluation metric, whether the tweet was submitted in the morning or after noon. 
I created a variety of testing functions to try and optimize my tree parameters, like picking a good ccp\_alpha and checking if any of my input columns would improve the model if they were removed. These were fun because they helped me put the parameters into perspective, since I could change them and directly see impacts on my model. 
MY decision tree was really accurate, with a mean score of 81% from the cross validation, probably because I was more familiar with the kind of data it contained and really able to control it and pick the most powerful predictors (for example, coming from a historically red state or a historically blue state). 

## Part 3: Making my SVM
My SVM was not super promising, its cross validation scores were mostly under 70%. I think that experimenting with assigning numerical values to my categorical variables was still a really cool way to experiment with this data and really showcased the flexibility of machine learning models. Working with tokenizing my data and streamling the sklearn tools by removing redundant functions was a great way to better understand the details of how this model works. Rather than using CountVector and TfidfTransformer separatemly, I found a function called TfidfVectorizer that did both. 
Working through this classification task taught me a lot about how to think through & better process my information and break it down into parts.  
 
 ## Value Add Statement

I used resources from scikit-learn's documentation to create my text based SVM model (1). 
I read their  tutorial on working with text data and SVM's to learn what tools I needed to tokenize 
my text, and how to structure my code (what steps I needed to process the text, which formulas
I could choose from, what order they needed to go in..) 
I then went to the specific documentation for those functions (CountVectorizer, SGDClassifier..) 
to read the exact parameters available to me and modified the examples I was given to work best 
with my data and question. 

For the CountVectorizer function, I added the parameters 'strip_accents = "unicode", stop_words = 
{'english'}, and min_df=0.0001' to my function after a bit of experimentation. Because my data
was made up of tweets, I wanted to make sure I was removing things like emojis that got turned
into meaningless strings in the spreadsheets. I also wanted to remove english stop words. One
of the things the text based SVM takes into account is frequency of words, and I didn't want to 
muddy that data with common but ultimately meaningless words like "the" and "a" that would show
up super frequently. The final parameter change I made was adding a min df, which made it so words 
that appeared in less than .01% of the dataframes were filtered out. This helped my model run a bit 
faster, and also kept things like links from clouding up my analysis. 

I tried out a few different classifier options, and ended up settling on the SGDClassifier. It gave
more accurate predictions, particularly when I set the loss system to "squared error", which entails
a linear regression (2). I didn't feel like the penalty parameters and some of the other things 
for this model really applied to the question I was trying to answerm so I left them out.

I also used scikit-learn's documentation to find a way to automate my cross validation (5). I used
this guide to find the cross_val_score function and import, and then read the documentation on it
to select the parameters I felt were a good check (cross validation of 10, accuracy based scoring).
I also took the print statement from this guide ("%0.2f accuracy with a standard deviation of 
%0.2f") because I thought it was much nicer to look at than just printing the scores as I was doing.
I also had to mess aroung with my data to make it work with this model. I found a nifty bit of code 
on stack overflow to turn my True/False party column into 1s and 0s that I took and adapted for my 
code (4), I needed it to make my crossval check work , since the cross_val_score function could not 
compare the True/False to the binary tree output like my original cheap check (np.mean == 
tree_test_df.party). I thought using a dictionary and list iteration was a super efficient way
of indexing and changing values.


I adapted a lot of my tree code from Lab 16. I wrote some stuff to adapt my data into a binary setup
that was compatible with the Lab 16 functions and things ot explore my data with. I also had to 
adapt all of the code I took to work with my data, or vice versa. In the SVM, I added some slicing
to my test data when I transform it, because the word tokens aren't quite what was expected. I had
to adapt some of my variables for both functions, switching their encoding and removing extra 
stuff like 3rd party candidates. 

I took my data from fivethirtyeight on github (6).

1. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
2. https://scikit-learn.org/stable/modules/sgd.html
3. Lab 16, from this course
4. https://stackoverflow.com/questions/51016230/how-to-change-values-in-a-column-into-binary
5. https://scikit-learn.org/stable/modules/cross_validation.html 
6. https://raw.githubusercontent.com/fivethirtyeight/data/master/twitter-ratio/senators.csv

