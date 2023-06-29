# Machine Learning System Design
Let’s say you want to build a spam classifier . Here are some obvious examples of spam and non-spam emails.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5692026c-a4d9-48a9-baae-4ab63507a99e)

Let’s say we have a labeled training set of some number of spam emails and some non-spam emails denoted with labels y equals 1 or 0. How do we build a classifier using supervised learning to distinguish between spam and non-spam?

In order to apply supervised learning, the first decision we must make is how do we want to represent x i.e. features of the email. Given the features x and the labels y in our training set, we can then train a classifier, for example using Logistic Regression .

Here is one way to choose a set of features for our emails. We could come up with say, a list of may be a hundred words that we think are indicative of whether email is spam or non-spam. E.g. deal, buy, discount et.c indicate spam, Andrew, now, etc. indicate non-soam,…

Given a piece of email, we an then take this piece of email and encode it into a feature vector as follows:

We take our list of 100 words and sort them in alphabetical order:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/09e45a63-c49c-43ef-a3d3-c167d259c641)

Each of our features x<sub>j</sub> will basically be 1 if word j appears in the email and x<sub>j</sub> will be 0 otherwise.

Note: In practice, take most frequently occurring n words (10,000 to 50,000) in training set, rather than manually pick 100 words.

## Building a Spam Classifier
How to spend your time to make it have low error?
	- Collect lots of data
		- E.g. “honeypot” project
		- We have seen previously getting lots of data often helps but not all the time.
	- Develop sophisticated features based on email routing information (from email header)
		- When spammers send email, very often they will try to obscure the origins of the email, and may be use fake email headers or send email through very unusual sets of computer service. Through very unusual routes, in order to get the spam to you. And some of this information will be reflected in the email header.
	- Develop sophisticated features for message body, e.g. should “discount” and “discounts” be treated as same word? How about “deal” and “dealer”? Features about punctuation.
	- Develop sophisticated algorithm to detect and correct misspellings (e.g. m0rtgage, med1cine, w4tches.). Spammers actually do this because if you have watches with a 4 in there then with the simple technique that we talked about just now, the spam classifier might not equate  this as a same thing as the word “watches” and so it may have harder time realising that something is spam with these deliberate misspellings.## Building a Spam Classifier
How to spend your time to make it have low error?
	- Collect lots of data
		- E.g. “honeypot” project
		- We have seen previously getting lots of data often helps but not all the time.
	- Develop sophisticated features based on email routing information (from email header)
		- When spammers send email, very often they will try to obscure the origins of the email, and may be use fake email headers or send email through very unusual sets of computer service. Through very unusual routes, in order to get the spam to you. And some of this information will be reflected in the email header.
	- Develop sophisticated features for message body, e.g. should “discount” and “discounts” be treated as same word? How about “deal” and “dealer”? Features about punctuation.
	- Develop sophisticated algorithm to detect and correct misspellings (e.g. m0rtgage, med1cine, w4tches.). Spammers actually do this because if you have watches with a 4 in there then with the simple technique that we talked about just now, the spam classifier might not equate  this as a same thing as the word “watches” and so it may have harder time realising that something is spam with these deliberate misspellings.

While working on a Machine Learning problem, very often you can brainstorm lists of different things to try, like these. It would be very hard to tell out of these 4 options which is the best use of your time.

So what happens far too often is that a research group or product group will randomly fixate on one of these options and sometimes that turns out not to be the most fruitful way to spend your time depending upon which of these options someone ends up randomly fixating on.

# Error Analysis
If you’re starting work on a Machine Learning problem or building a machine learning application, it’s often considered very good practice to start, not by building a very complicated system with lots of complex features and so on. Below is the recommended approach:

- Start with a simple algorithm that you can implement quickly. Implement it and test it on your cross-validation data.
- Plot learning curve to decide if more data, more features, etc. are likely to help.
- Error Analysis: Manually examine the examples (in cross validation set) that your algorithm made errors on. See if you spot any systematic trend in what type of examples it is making errors on.

## Here is a specific example:
m<sub>cv</sub> = 500 examples in cross validation set. Let’s say algorithm misclassifies 100 emails.
Manually examine 100 errors, and categorise them based on:
- What type of email it is e.g. Pharma (trying to sell drugs), replica (trying to sell replica of watches etc.), steal passwords, ..
    - Let’s say we came up with, Pharma = 12, Replica = 4, Steal Passwords = 53, Others = 31. By counting up number of email in these different categories you might discover e.g. that the algorithm is doing particularly poorly on emails that are trying to steal passwords and that may suggest that it might be worth your effort to look more carefully at that type of email and see if you can come up with better features to categorise them correctly.
- What cues (features) you think would have helped the algorithm classify them correctly.
    - Lets say some of our hypothesis about things or features that might help us classify emails better are: trying to detect 
        - deliberate misspellings 
        - versus unusual email routing 
        - versus unusual spamming punctuations.
            - Let’s say we found 5 cases of deliberate misspelling, 16 of unusual email routing and 32 of unusual spamming punctuations and bunch of other type of emails, And this is what you get on your cross validation set then it really tells you that may be deliberate spelling is sufficiently rare phenomenon and may be not worth all the time trying to write algorithms that detect that. But If you find lots of spammers are using unusual punctuation then may be that’s the strong sign that it might actually be worth your while to spend the time to develop more sophisticated features based on the punctuation.

# The importance of numerical evaluation
When developing learning algorithms, one other useful tip is to make sure that you have a numerical evaluation of your learning algorithm.

Here is a specific example:
- Let’s say we are trying to decide whether or not we should treat words like discount, discounts, discounted, discounting as a same word?
    - So you know may be one way to do that is to just look at the first few characters in the word like, you figure out all these words have roughly the similar meaning.
    - In Natural Language Processing the way this is done is actually using a type of software called Stemming Software. And if you ever want to do this yourself, search on a web search engine for the “Porter Stemmer” and that would be one reasonable piece of software for doing this sort of stemming, which will let you treat all these words as the same word.

But using a stemming software that basically looks at first few alphabets of the word, more or less it can help, but it can hurt. And it can hurt because for example, the software may mistake the word “universe” and “university” as being the same thing.

So if you are trying to decide whether or not to use stemming software for a spam classifier, it’s not always easy to tell and in particular Error Analysis may not actually be helpful for deciding if this sort of stemming idea is a good idea.

Instead the best way to figure out if using stemming software is good to help your classifier is if you have a way to very quickly just try it and see if it works. And in order to do this having a way to numerically evaluate your algorithm is going to be very helpful.

Concretely, may be the most natural thing to do is to look at the cross validation error of the Algorithm’s performance with and without stemming. 

For this particular problem there is a very natural, single real number evaluation metric namely the Cross Validation Error. 

One more quick example, let’s say you also trying to decide whether or not to distinguish between upper vs lower case?
Once again because we have a way to evaluate our algorithm we can try to evaluate our algorithm with or without differentiating and decide based on the Cross Validation error.

So when you are developing a learning algorithm, very often you’ll be trying out lots of new ideas and lots of new versions of your learning algorithm. If every time you try out a new idea, if you end up manually examining a bunch of examples again to see if it got better or worse, that’s gonna make it really hard to make decisions on. Do you use stemming or not? Do you distinguish upper and Lowe case or not?

But by having a single real number evaluation metric, you can then look and see if the error went up or down and you can use that much more rapidly to try out new ideas and almost right away tell if your new idea has improved or worsened the performance of the learning algorithm. This will let you often make much faster progress. 
