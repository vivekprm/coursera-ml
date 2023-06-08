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
