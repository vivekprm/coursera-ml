Our first learning algorithm will be **Linear Regression**.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/989182a1-eac2-4353-82d3-e96f648e5691)

It's a Linear Regression Problem. We need a set of training set data to make predictions:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1f1bf1f8-afa3-4bcc-879b-feeca6d9b043)

To establish notation for future use, we’ll use x<sub>(i)</sub> to denote the “input” variables (living area in this example), also called input features, 
and *y*(*i*) to denote the “output” or target variable that we are trying to predict (price).

A pair **(*x*(*i*),*y*(*i*))** is called a training example, and the dataset that we’ll be using to learn—a list of **m training examples**(x(i),y(i)); i=1,…,m — is called a training set.

Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, X = Y = ℝ.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

![](:/15508a516f0f0a9b3ff8e48e6e2c24aa)

When the target variable that we’re trying to predict is **continuous**, such as in our housing example, **we call the learning problem a regression problem**. When y can take on only a **small number of discrete values** (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a **classification problem**.

**Cost Function**

We'll define something called the cost function, **this will let us figure out how to fit the best possible straight line** to our data.

![](:/c5edcd7274ec7816784c233973216966)

Our straight line looks as below for different values of 𝛉0, 𝛉1:

![](:/42c708fa505a3d5f064c12d76ce993de)

Idea: Choose 𝛉0, 𝛉1 such that h𝛉(x) is close to y for our training examples (x, y):

Minimize (h𝛉(x) -y)2 for 𝛉0, 𝛉1. For the whole training sets.

m                                                                    m

Σ (h𝛉(x) -y)2. Should be minimised. Or, 1/2m Σ (h𝛉(x) -y)2 should be minimum. Which makes math little easier.

I=1                                                                 I = 1

Which is the cost function:
                           m
J(𝛉0,  𝛉1) = 1/2m Σ (h𝛉(x) -y)2
                           I = 1

Minimize J(𝛉0, 𝛉1). Cost function is also called squared error function.

![](:/96ceead7017d776373c17ffde3b39adf)

**Cost Function Institution 1**

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by h𝛉(x)) which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of J(*θ*0,*θ*1) will be 0. The following example shows the ideal situation where we have a cost function of 0.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1579996800000&hmac=nmFpqdBk2bzPYeWDf3HStcoffC1rQGYk51Be3mohIps)

When *θ*1=1, we get a slope of 1 which goes through every single data point in our model. Conversely, when *θ*1=0.5, we see the vertical distance from our fit to the data points increase.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1579996800000&hmac=qF6W-KNAxUxypvNfSQkk_DB5wPhLpvm0OJOrk_exp6g)

This increases our cost function to 0.58. Plotting several other points yields to the following graph:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1579996800000&hmac=ScPtXbdHURQm1RqnblLIiOe4n3Cx80BtZWpiU1GCDqI)

Thus as a goal, we should try to minimize the cost function. In this case, *θ*1=1 is our global minimum.

### Cost Function - Intuition II

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1579996800000&hmac=HwUxuv1cO7rQxn2evVl5ZYrfhG60DgnZL25tafM7u-Y)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for J(*θ*0,*θ*1) and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when *θ*0 = 800 and *θ*1= -0.15. Taking another h(x) and plotting its contour plot, one gets the following graphs:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1579996800000&hmac=7HjvlqtpXxG1V5BpzS2zgxCwKN7Z7iihWQkFxlboyHc)

When *θ*0= 360 and *θ*1 = 0, the value of J(*θ*0,*θ*1) in the contour plot gets closer to the center thus reducing the  cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1579996800000&hmac=MjWhr1tGrBX_fXlmjO_yMYIwsPLXEBLuS7s0q41Kewk)

The graph above minimizes the cost function as much as possible and consequently, the result of *θ*1 and *θ*0 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

# Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields *θ*0 and *θ*1 (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put *θ*0 on the x axis and *θ*1 on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1580169600000&hmac=NmvPelVmbfV2a2ZoLYUNkwsCzRC49TxC3kSPj5QEXF8)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. **The slope of the tangent is the derivative at that point and it will give us a direction to move towards**. We make steps down the cost function** in the direction with the steepest descent**. The size of each step is determined by the parameter α, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of J(*θ*0,*θ*1). Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

repeat until convergence:

**θj:=θj−α * ∂/∂θj J(θ0,θ1)**

where

j=0,1 represents the feature index number.

At each iteration j, one should simultaneously update the parameters *θ*1,*θ*2,...,*θn*. Updating a specific parameter prior to calculating another one on the *j*(*th*) iteration would yield to a wrong implementation.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56.png?expiry=1580169600000&hmac=sfw0cFQa4GFFJLfbKtLNbWDUDEG39lKIOk7mNyhyzF8)

# Gradient Descent Intuition

We explored the scenario where we used one parameter *θ*1 and plotted its cost function to implement a gradient descent. Our formula for a single parameter was :

Repeat until convergence:

***θ*1:=*θ*1−*α * d/dθ*1*J*(*θ*1)**

Regardless of the slope's sign for *d/dθ*1*J*(*θ*1), *θ*1 eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of *θ*1 increases and when it is positive, the value of *θ*1 decreases.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/SMSIxKGUEeav5QpTGIv-Pg_ad3404010579ac16068105cfdc8e950a_Screenshot-2016-11-03-00.05.06.png?expiry=1580169600000&hmac=uXi-4psGAJ3DglGtKnDOKNpcJGy81OIa12Q5jFWj0Qk)

On a side note, we should adjust our parameter *α* to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/UJpiD6GWEeai9RKvXdDYag_3c3ad6625a2a4ec8456f421a2f4daf2e_Screenshot-2016-11-03-00.05.27.png?expiry=1580169600000&hmac=AZMaBe3K8cK60dEhua_2JF3YCP9YDFa2LJm5IQ90fsw)

###

### How does gradient descent converge with a fixed step size *α*?

The intuition behind the convergence is that *d*/*dθ*1*J*(*θ*1) approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:

***θ*1:=*θ*1 − *α*∗0**

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00.png?expiry=1580169600000&hmac=eI2Y3k48Knm_0Vndz7i6J2w60mW8f8Lrk8iJX9uIuNw)

### Gradient Descent For Linear Regression

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

repeat until convergence: {
*θ*0 := *θ*0−*α*1*m*∑*i*=1*m*(*hθ*(*xi*)−*yi*)
*θ*1:= *θ*1−*α*1*m*∑*i*=1*m*((*hθ*(*xi*)−*yi*)*xi*)
}

where m is the size of the training set, *θ*0 a constant that will be changing simultaneously with *θ*1 and *xi*, *yi  *are values of the given training set (data).

Note that we have separated out the two cases for *θj* into separate equations for *θ*0 and *θ*1; and that for *θ*1 we are multiplying *xi  *at the end due to the derivative. The following is a derivation of ∂/∂*θjJ*(*θ*) for a single example :

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/QFpooaaaEea7TQ6MHcgMPA_cc3c276df7991b1072b2afb142a78da1_Screenshot-2016-11-09-08.30.54.png?expiry=1580169600000&hmac=AcBdnmYkLl8s6QQhdl9JHWJP3T6AsGATH-K0qsSzHtQ)

The point of all this is that** if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.**

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus **gradient descent always converges (assuming the learning rate α is not too large) to the global minimum**. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1580169600000&hmac=eF_II4IgAvoWQAD__tjM3045rCNcoDknXB0lZ2aYx6M)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.
