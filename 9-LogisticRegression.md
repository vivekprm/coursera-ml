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

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/be568dd6-0b53-4182-990e-e13cfaccc685)

So, you can see it asymptotes at 1 and 0.

# Interpretation of Hypothesis Output
h<sub>θ</sub>(x) = estimated probability that y = 1 on input x

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0c027fc3-d947-4025-90c5-712b59230cb5)

Since h<sub>θ</sub>(x) outputs 0.7, Tell patient that there is 70% chance of tumor being malignant.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/296a71e8-c1de-440b-acdd-8d59ae313147)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b2d08f31-d8ce-4b77-aba7-d4c68a2e2b2a)

# Decision Boundary
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3d9d1e77-d286-45e2-ad9e-2dcb02b10249)

From the plot of sigmoid function we can see that g(z) is >= 0.5, whenever z >= 0

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/596f2d5d-c93d-4dc7-b631-c84c56a807cb)

Similarly g(θ<sup>T</sup>x) < 0.5 when θ<sup>T</sup>x < 0

Let's use this to better understand how the hypothesis of logistic regression makes those predictions. Let's suppose we have training set and hypothesis like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9034771b-f8f5-4a9f-adae-d4394e920772)

Lets suppose we endup choosing θ<sub>0</sub> = -3, θ<sub>1</sub> = 1 & θ<sub>2</sub> = 1. So our parameter vector is going to be:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9331302e-0cc6-44b4-90bd-de65d6689aee)

Given this choice of hypothesis parameters, let's try to figure out where a hypothesis would endup predicting y equals 1 and where it will end up predicting 
y equals 0.

Predict y = 1 if -3 + x<sup>1</sup> + x<Sup>2</sup> >= 0

In this case θ<sup>T</sup>x = -3 + x<sup>1</sup> + x<Sup>2</sup>
Or x<sup>1</sup> + x<Sup>2</sup> >= 3

So, x<sup>1</sup> + x<Sup>2</sup> = 3 is a line which is called Decision Boundary:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d0d86ace-844e-4470-801a-fecf8764215b)

Now let's look at more complex example. So given a training set like below how can we use Logistic Regression to fit this sort of data?

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/429e7087-ae4d-4e0e-8369-c65b3f54048d)

Let's say our hypothesis looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d718914e-ce31-4e01-aef9-bb6b439b3b1e)

Where we have added two extra features, x<sub>1</sub> squared and x<sub>2</sub> squared to my features. So now we have 5 parameters θ<sub>0</sub>, θ<sub>1</sub>,
θ<sub>2</sub>, θ<sub>3</sub>, θ<sub>4</sub>.

We will see how to choose values of these paramters. But for now Let's say θ<sub>0</sub> = -1, θ<sub>1</sub> = 0,
θ<sub>2</sub> = 0, θ<sub>3</sub> = 1, θ<sub>4</sub> = 1

So now our parameter vector looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/022c6f93-7ce9-4497-8eb1-4aa749b02a77)

And so now we have hypothesis as:
y = 1 if -1 + x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> >= 0
or x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> >= 1

So in this case Decision Boundary looks like a circle of radius 1.

Decision Boundary is a property, not of the training set but of the hypothesis under the parameters. So long we are given our parameter vector theta that defines the
Decision Boundary. But the training set is not what we use to define the Decision Boundary.
The training set may be used to fit the parameters theta (we'll see that later). But once you have the paramters theta, that is what defines the decision boundary.

We can come up with much complex decision boundaries by adding much higer order polynomial terms.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/57fab35e-2f69-491e-a66c-8207c24e73ad)

# Logistic Regression - Cost Function
Here is the supervised learning problem of fitting logistic regression model

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ca21d08e-53b4-43c8-9659-c8316fd97014)

When we were using Linear Regression we used the following Cost Function:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/36107dac-f800-4121-9967-d4759d560802)

And alternative way of writing this cost function 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8008e767-32fc-404b-b3df-c232c94a3152)

To simplify this equation bit more we can get rid of those superscripts i. Here h<sub>θ</sub>(x) is the prediction and y is actual value.

This cost function worked fine for Linear Regression but here we are interested in Logistic Regression. If we could minimize this cost function J here that will
work ok. But it turns out that if we use this particular cost function, this would be a non convex function of the parameter's data.

If we take the sigmoid function and replace h<sub>θ</sub>(x) in the cost function we get plot for J(θ) similar to below, with many local minimum:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c2b1095f-19fe-4b93-812a-821e79d86f91)

In contrast we would like to have a cost function J(θ) that is convex, so that when we run Gradient Descent we would be guaranteed to descend and converge to 
the Global minimum.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/7c03fcc1-cd5d-4d3d-bdde-14be52dd3f72)

And the problem with using this cost function is that because of this very non-linear Sigmoid function that appears in place of  h<sub>θ</sub>(x), J(θ) ends up being 
a non-convex function. So we will comeup with a different Cost Function that is Convex.

Here is the cost function for Logistic Regression

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/658dcbbb-9808-408b-8003-841f7a8fcbda)

It's plot looks like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b806e9ca-5864-42f9-88ab-3736bc5895f8)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c2b1095f-19fe-4b93-812a-821e79d86f91)

This Cost Function has few interesting and desirable properties:
- If y = 1, h<sub>θ</sub>(x) = 1 i.e. prediction is same as actual value then the Cost = 0. Which is intersection point on the x axis in the graph.
- As h<sub>θ</sub>(x) -> 0, Cost -> ∞
  - Captures the intuition that if h<sub>θ</sub>(x) = 0 ( Predict  P(y = 1 | x;θ) = 0), but y = 1. We will penalize learning algorithm by a very large cost if it turned out to be wrong! As we are saying probability of y = 1 is zero  when h<sub>θ</sub>(x) = 0. And if we turned out to be wrong penalty is huge. Cost -> ∞ signifies that.

Now let's look at the plot when y = 0

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/eeec6e7d-244f-4d8f-95e4-8d451c32f18b)

# Simplified Cost Function and Gradient Descent
Here is our Cost function for Logistic Regression

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5b9ef79b-4e87-4662-bae9-424a195052d1)

Because y is 0 or 1 will comeup with simpler cost function. We can compress above equation into one equation and we will comeup with more convenient cost function.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ddd5dd89-2a14-4621-8ebd-8fbed4cf3349)

A vectorized implementation is:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c94b0488-4cbc-42fb-ae7a-ed1ff47ddaf9)

# Advanced Optimization
"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value θ:
- J(θ)
- ![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ad5ead4c-ef95-493d-95df-742fea52a895)

We can write a single function that returns both of these:
```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

# Multiclass Classification : One vs All
Here are some examples:
- Email Foldering/Tagging: Work, Friends, Family, Hobby
- Medical Diagrams: Not ill, Cold, Flu
- Weather: Sunny, Cloudy, Rain, Snow

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/829183e6-7349-4194-9718-5b308a0c0b94)

How do we get a learning algorithm to work for this setting?
We can use one-vs-all and make it work for multiclass classification problem. Here is how One-vs-All classification works
What we are going to do is take a training set and turn this into three separate Binary Classification problem.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9b7e7ae8-16ac-43cf-9d0a-594bd7aaa195)

In summary we have fit three classifiers i = 1, 2, 3
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/52058e6a-7a2b-4ce4-ab88-158fb6b5fe8b)

In summary, Train a Logistic Regression classifier h<sub>θ</sub><sup>(i)</sup>(x) for each class i to predict the probability that y = i
On a new input x, to make a prediction, pick the class i that maximizes ![image](https://github.com/vivekprm/coursera-ml/assets/2403660/64633fdf-9de7-4bea-be1b-2d764ebed769)

# Solving The Problem of Overfitting
Lets look at example of our predicting housing prices with Linear Regression.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3154d2e7-fb45-4ee6-b8ca-a7bccc8426d3)

So last hypothesis fits all the training examples but not very good for hoursing prices as it has very high variance. Second fitting is just right.

Overfitting: If we have too many features, the learned hypothesis may fit the training set very well (J(θ) ~ 0) but fail to generalize to new examples (predict prices on new examples)

Similar thing can be said about Logistic Regression as well. Here is an example

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/796992b8-9d8b-41f6-8dd1-cc847e759df2)

## Addressing Overfitting
We might have a learning problem which might have lots of features so plotting the hypothesis may not work. If we have lots of features and ery less training data then ovefitting can be a problem. In order to address overfitting we have two main options that we can do:

- Reduce number of features
  - Manually select which features to keep.
  - Model Selection algorithm (later in the course)
    - Algorithm to automatically decide which feature to keep and which to discard
  - But may be we might need all the feature and all of them are important.
- Regularization
  - Keep all the features, but reduce magnitude/values of parameters θ<sub>j</sub>
  - Works well when we have lot of features, each of which contribute a bit to predicting y.   

# Regularization : Cost Function
## Intuition
We see that quadratic function gives pretty good fit to the data. Wereas if you fit to a overly high order polynomial, you end up with a curve that fits the training set very well but doesn't generalize well.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c7a96247-a6a2-4d49-82eb-92e6313e5c1c)

Suppose we penalise and make θ<sub>3</sub>, θ<sub>4</sub> really small.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/fa6cc2af-20fe-4e5b-aebc-338d8a8a6933)

Our objective is to minimize this cost function. Now let's add below terms in our cost function:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/63a6603f-4239-439e-bdfb-d2a59269f192)

Now only way to minimize this new cost function is only if θ<sub>3</sub>, θ<sub>4</sub> are small. So to minimize this new cost function θ<sub>3</sub> ~ 0 and θ<sub>4</sub> ~ 0. So we will be basically left with quadratic funtion.

Here is the idea behind Regularization:
Small values for parameters θ<sub>0</sub>, θ<sub>1</sub>, ...., θ<sub>n</sub>
- Leads to simpler Hypothesis
- Less prone to overfitting

Housing
- Features: x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>,....., x<sub>100</sub>
- Parameters: 0<sub>0</sub>, 0<sub>1</sub>, 0<sub>2</sub>, ......, 0<sub>100</sub>

So we have set of 100 features it's hard to pick in advance which are the ones that are less likely to be relevant. So we have 101 parameters and we don't know which ones to pick to try to shrink.

So in Regularization what we are going to do is, take our cost function (below for Linear Regression), and modify our cost function to shrink all of our paramters. Because we don't know which one to shrink.
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/328d769f-30c0-4e5c-afc4-4ddae777c242)

So we are going to modify our Cost Function and add a term (Regularization Term) in the end as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6816eacb-2b54-41dd-9a48-a830e3174a40)

Notice that we are not penalizing 0<sub>0</sub>. λ here is called Regularization Parameter, which controls a tradeoff between two different goals:
- We would like to fit the training data well which is captured by first term.
- We want to keep the parameters small which is captured by the second term.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/93c969ab-510d-4630-9024-86d5c3b6a197)

In Regularized Linear Regression if λ is set to an extremely large value (perhaps far too large for our problem, say λ = 10<sup>10</sup>. So what will happen, we will end-up penalizing 0<sub>1</sub>, 0<sub>2</sub>, 0<sub>3</sub>, 0<sub>4</sub> and we will endup with all these parameters close to zero. If we do that it's like we are getting rid of these terms in our hypothesis. So we are just left with h<sub>θ</sub>(x) = 0<sub>0</sub>, which is horizontal straight line and this is an example of underfitting.

# Regularized Linear Regression
So we will take Gradient Descent and Normal Equation and generalize them using Regularization.

## Gradient Descent
Old equation

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ae889dae-e860-4821-b5dc-d0c77d5421c6)

All we need to do is modify the second equation as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9a1075cf-5c45-42fe-a496-0c0fdc374dca)

The term (1 - α * (λ/m)) is going to be a number < 1. Because α * (λ/m) is going to be positive. α * (λ/m) < 1 So we are srinking the 0<sub>j</sub> with every iteration. And Second term is exactly the same as in original Gradient Descent.

## Normal Equation with Regularization

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f3f40da5-dcbd-41c8-87bf-6c728aabd2d0)

### Non-invertability (optional / advanced)
Suppose m <= n

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f7d2ba5e-cce2-4283-93a0-3213ead6f386)

Even if X<sup>T</sup>X is non-invertible, regularization makes it invertible so long as λ > 0.

# Regularized Logistic Regression
We saw that Logistic Regression is prone to overfitting with higher order polynomial like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e51d12d1-6597-49e7-ab95-b1bf89b2b53c)

Now if we want to modify it and use Regularization all we need to do it add to it following:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e3496f9d-4bff-4577-bfe0-88daaadd41a7)

## Gradient Descent
Normal gradient descent without regularization

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/41670564-6414-4ebc-bd75-b118ab745bb6)

To use regularization as in Linear Regression modify the second equation as follows

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e991d1e7-6f5f-457e-b25d-9ba81da1222a)

## Advanced Optimization

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d5a5192f-f32c-4cbf-a6b9-69e0287513c7)

With Regularization it looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/98ffd981-b17d-4e0e-9f30-8d2e02347b88)



