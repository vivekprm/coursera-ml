# Classification
Here are some of the classification problems:
- Email: Spam / Not Spam
- Online Transactions: Fraudulent (Yes / No)?
- Tumor: Malignant / Benign

In all these problems the variable that we are trying to predict is taking two discrete values, also called Binary Classification Problem:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/84c2063b-3e04-41e7-a3c1-89a59f13a3a5)

We can also have multi class problems where variable y can take multiple values e.g. 0, 1, 2, 3 etc. It is called Multiclass Classification Problem.

Here is the training set for a problem:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/4d972818-82e6-417a-9d0f-672047a211c9)

We can try and fit linear regression on this dataset:
h<sub>θ</sub>(x) = θ<sup>T</sup>x

And you may get hypothesis like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a58ac225-bb68-44c9-9b57-7a4fdeeb2b00)

If you want to make prediction. One thing you could try doing is then threshold the classifier output at 0.5:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/644e65a7-9e55-4eb1-a477-28967a8e6230)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e7670ea0-69ea-442e-9f2e-f8d8d0480069)

It looks like linear regression is doing something reasonable in this case eventhough it is a classification task. But now let's try changing the problem a bit.
Let's extend horizontal axis a little bit and let's say we got one more training example way out right as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/bff9e530-8dd3-4e09-ac86-82735ccec2b4)

So now if we run linear regression it might have a fit like blue line and threshold is changed which looks bad and predictions are not good.

So applying linear regression to a classification problem often isn't a great idea. In the first instance before we added extra training example Linear Regression
just got lucky.

Here is one more funny thing what would happen if we were to use Linear Regression for a classification problem.
For Classification we know y is either 0 or 1. But if we are using Linear Regression, hypothesis hsub>θ</sub>(x) can output values much larger than 1 or less than 0
even if all our training examples have labels y = 0 and y = 1

So now we will develop an algorithm called Logistic Regression which has the property 0 <= h<sub>θ</sub>(x) <= 1

Eventhough it has regression in the name Logistic Regression is classification algorithm.

# Hypothesis Representation
So now we are going to change our hypothesis of linear regression as below:

h<sub>θ</sub>(x) = g(θ<sup>T</sup>x)

Where:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/110902e1-6000-4510-badb-59631d8022cd)

This is called Sigmoid Function or the Logistic Function. Alternative way of writing the hypothesis function:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c0e5c909-a888-4f5a-a1dd-a49109ca3986)

Lastly let's see how Sigmoid function looks like:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2d6e80ca-1a7e-48ae-8193-1dadf27a5e9f)

SO you can see it asymptotes at 1 and 0.

