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

# Handling Skewed Data
There is one important case, where it’s particularly tricky to come up with an appropriate error metric or evaluation metric for our learning algorithm. That case is the case of what’s called Skewed Classes.

Consider Cancer Classification Example:
We have features of medical patients and we want to decide whether or not they have cancer. This is like malignant vs benign tumour problem.

Train Logistic Regression Model h<sub>θ</sub>(x). (y = 1 if cancer, y = 0 otherwise)

Let’s say be test our classifier on a test set and find that we get 1 percent error. So, we are making 99% correct diagnosis. Seems like a really impressive result right?

But now let’s say we found out that only 0.50% patients in our training and test set actually have cancer. In this case 1% error no longer looks so impressive .

```
function y = predictCancer(x)
	y = 0; 	%ignore x!
return
```

Here is a piece of code, here is actually a piece of non-learning code that takes this input of features x and it ignores it and it just sets sets y equals 0 and always predicts, you know, nobody has cancer and this algorithm would actually give 0.5% error. So this is even better than the 1% error that we were getting just now and this is a non-learning algorithm which is predicting y = 0 all the time.

In this case the number of positive examples is much much smaller than the number of negative examples because y=1 is so rarely, this is what we call case of Skewed Classes. We will have lot more examples from one class than the other class.

So the problem with using classification error or classification accuracy as our evaluation metric is the following.

Let’s say you have one learning algorithm that’s getting 99.2% accuracy, so that’s 0.8% error. Let’s say you make a change to your algorithm and you now are getting 99.5% accuracy I.e 0.5% error. So is this an improvement to algorithm or not?

One of the nice things about having a single real number evaluation metric is this helps us to quickly decide if we just need a good change to the algorithm or not? By going from 99.2% to 99.5% accuracy, did we do something useful or did we just replace our code with something that just predicts y = 0 more often. So if you use skewed classes it becomes much harder to use just classification accuracy because you can get very high classification accuracies or very low errors and it’s not always clear if doing so is really improving the quality of your classifier. Because prediction y=0 all the time doesn’t seems like a good classifier.

When we face such skewed classes therefor we would want to come up with different error or evaluation metric. One such evaluation metric are what’s called Precision/Recall.

## Precision/Recall
Let’s say we are evaluating a classifier on a test set, for the examples in the test set the Actual Class of that example in the test set is going to be either 1 or 0 if this is a Binary Classification problem.

What our learning algorithm will do, it will predict some value for the class and our learning algorithm will predict value for each example in my test set and predicted value will also be either 0 or 1. So let’s draw a 2 x 2 table as follows:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0c9d1ff7-f92c-47b7-8c54-46a1a07c4206)

Here is a different way to evaluate performance of our algorithm, we are going to compute two numbers:

### Precision
Of all the patients where we predicted y=1 , what fraction actually have cancer?
i.e. No of True Positives / No of Predicted Positive

Another way to write this would be:
Number of True Positives / Number of (True Positives + False Positives)

You can tell high precision will be good.

### Recall
Of all the patients that actually have cancer, what fraction did we correctly detect as having cancer.

Number of True Positives / Number of Actual Positive

Another way to write this would be:

Number of True Positives / Number of (True Positives + False Negative)

So having a high recall would be a good thing.

So by calculating precision and recall this will gives us better sense of how well our classifier is doing. 
So if our y=0 classifier will have a recall equal to 0 because there won’t be any true positives. So it’s not a very good classifier.

So a classifier with high precision and high recall is a good classifier and we can’t cheat by always predicting y=0 because in such case recall will be 0.

## Trading off Precision & Recall
For many applications we want to somehow control the trade-off between precision and recall. Here are the definition of Precision and Recall

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/bd3bdad0-8c1c-4908-ba3d-739bf074d497)

Suppose we want to predict that the patient has cancer only if we’re very confident that they really do. One way to do this would be to modify the algorithm, so that instead of setting this threshold at 0.5, we might instead have 0.7. 

So we are saying someone has a cancer only if we think that there is greater than equal to 70% chance that they have cancer. So now you endue with a classifier which has higher precision but in contrast this classifier will have low recall because now we are going to predict y=1 for smaller number of patients.

Now consider a different example, suppose we want to avoid missing too many actual cases of cancer, so we want to avoid false negatives in particular if a patient actually has cancer, but we fail to tell them that they have cancer then that can be really bad. Because if we tell a patient that they don’t have cancer then they are not going for treatment which is not good. In such case rather than setting higher threshold value we might instead set it to a lower value e.g. 0.3. 

In this case we will have higher Recall and lower Precision.

So in general, for most classifiers there is going to be a tradeoff between precision and recall and as you vary the threshold, you can plot a curve that trades off precision and recall.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/73bc0a7e-6242-425d-a486-09a2bf93a9ac)

There can be many possible curves for precision and recall tradeoff.

This raises another interesting question: Is there a way to choose this threshold automatically? Or more generally, if we have a few different algorithms or a few different ideas for algorithms, how do we compare different precision recall numbers? 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/dbde6bb3-1306-4415-8853-80012b25b5e3)

How can we get a single real number evaluation metrics as it’s hard to figureout which algorithm is better using two metrics and time consuming as well?
One natural thing you might try is to look at the average precision and recall I.e. average = (P + R)/2

But this turns out not to be such a good solution, because similar to the example we had earlier it turns out that if we have a classifier that predicts y=1 all the time, then if you do that you can get a very high recall, but you end up with a very low value of precision.

Conversely, if you have a classifier that predicts y equals 0, almost all the time, that is it predicts y=1 very sparingly, this corresponds to setting a very high threshold using the notation of the previous y.  Then you can actually endup with a very high precision with a very low recall.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/842120a9-d1e2-4223-9846-9f355092df41)

So the two extremes of either a very high threshold or a very low threshold, neither of that will give a particularly good classifier and the way we recognise that is by seeing that we end up with a very low precision or a very low recall and if you just take the average from this example, the average is actually highest for Algorithm 3, even though you can get that sort of performance by predicting y=1 all the time and that’s just now a very good classifier. Algorithm 1 and Algorithm 2 would be more more useful than Algorithm 3.

So average of Precision and Recall is not a particularly good way to evaluate our learning algorithm.

In contrast, there is a different way for combining precision and recall, this is called the F score and it uses below formula:
F<sub>1</sub> Score = 2 * PR/(P + R)

So here are the F<sub>1</sub> Scores:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a47df1b6-cf4d-4692-89fb-21cc8cd27570)

We can see the Algorithm 3 has lowest F1 Score. And Algorithm 1 has the highest.

# Using Large Datasets
## Designing a High Accuracy Learning System
Two researchers Michelle Banko and Eric Broule ran the following fascinating study. They were interested in studying the effect of using different learning algorithms versus trying them on different training set sciences, they were considering the problem of classifying between confusable words. E.g. {to, two, too}, {then, than}

For breakfast I ate ____ eggs, should it be to or two or too?
In this case it’s two.

They took a few different algorithms, which were considered state of the art back in the day when they ran the study, they took the variance, roughly a variance on Logistic  Regression called the Preceptron. They also took some of the algorithms that were fairly used back then but somewhat less used now called Winnow and Memory-based. 

- Perceptron (Logistic Regression)
- Winnow
- Memory Based
- Naive Bayes

Exact algorithms here are their details are not important think of this as picking 4 different classification algorithms and really the exact algorithms are not important. But what they did was they varied the training set size and tried out these learning algorithms on the range of training set sizes and that’s the result they got (see below pic)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9e08aa36-7509-476e-ac89-98ad6554ba55)

And the trends are very clear right. First most of these algorithms give remarkably similar performance and second as the training set size increases, on the horizontal axis is the training set size in millions as you grow from hundred thousand upto a thousand million i.e. a billion training examples. The performance of the algorithms all pretty much monotonically increase and the fact that  you pick any algorithm may be pick a “inferior algorithm” but if you give that “inferior algorithm” more data, then from these examples it looks like, it will most likely beat even a “Superior Algorithm”.

So since this original study which is very influential, there’s been a range of many different studies showing similar results that show that many different learning algorithms, can sometimes, depending on the details can give pretty similar ranges of performance, but what can really drive performance is you can give the algorithm a ton of training data.

Results like these has led to a saying in machine learning that often in Machine Learning: 
"It’s not who has the best algorithm that wins. It’s who has the most data."

## Large data rationale
Assume feature x belongs to R<sup>n+1</sup> has sufficient information to predict y accurately.

Example: If we take the confusable words problems. Let’s say the feature x capture what are the surrounding words around the blank that we’re trying to fill in. So the features capture then we want to have, sometimes For breakfast I ate black eggs. Then that’s pretty much enough information to tell me that the word I want in the middle is TWO and that’s not word TO and it’s not the word TOO 

For breakfast I ate ____ eggs.

So the features capture, the surrounding words then that gives me enough information to pretty unambiguously decide what is the label y that we will be using to fill in the blank out of this set of three confusable words. 

Counter Example: Predict housing price from only size (feet<sup>2</sup>) and no other features.

Imagine I say house is 500 feet<sup>2</sup>, but we don’t give you any other feature, there are so many other factors that would affect the price of the house other than just the size of a house. So this is a counter example to this assumption that the features have sufficient information to predict the price to the desired level of accuracy.

Useful Test: Given the input x, can a human expert confidently predict y?

For the first example if we go to expert human English speaker, will be able to tell what word will go in the blank. So this gives us confidence that x allows us to predict y accurately.

But in contrast if we go to an expert realtor who sells houses for living, if we just tell them the size of the house won’t be able to tell the price of the house.

Let’s see when having lots of data helps:
Suppose the features have enough information to predict the value of y and Let’s suppose we use a Learning Algorithm with a large number of parameters. So may be Logistic Regression / Linear Regression with a large number of features, one thing that I usually do is using Neural Network with many hidden units that would be another learning algorithm with a lot of parameters. So these are all powerful learning algorithms with lot of parameters that can fit very complex functions. So we are going to call these low bias algorithms because you know we can fit very complex functions and because we have a very powerful learning algorithm that can fit very complex functions. Chances are, if we run these algorithms on a data set we will be able to fit the training set well hopefully the training error J<sub>train</sub>(θ) will be small.

Now let’s say we use a massive massive training set in that case even though we have lot of parameters but if the training set is sort of even much larger than the number of parameters then hopefully these algorithms will be unlikely to overfit which means J<sub>train</sub>(θ) ~ J<sub>test</sub>(θ)

Finally putting these two together, if the training set error is small and the test set error is close to the training error implies that test set error will also be small.

Another way to think about this is that, in order to have a high performance learning algorithm we want it to not have high bias and not have high variance. So the bias problem we are going to address by making sure our Learning Algorithm has many parameters and by using a very large training set, ensures that we don’t have variance problem either.
