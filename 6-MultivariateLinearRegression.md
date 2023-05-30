E.g. Instead of size of the house we can have many other features that can be used to predict the price of a house.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ec76ac1c-eb02-4e01-8ba2-d77645c9e3bb)

Notation:
n = number of features, in above case n = 4
x<sup>(i)</sup> = input (features) of i<sup>th</sup> training example. E.g in this case X<sup>(2)</sup> is:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b1ff073b-27b2-4084-bd16-0366e13f37fd)

x<sub>j</sub><sup>(i)</sup> = value of feature j in i<sup>th</sup> training example. E.g. X<sub>3</sub><sup>(2)</sup> = 2

# Hypothesis
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/15b8666b-4b3e-4468-b336-1f53f2a19d13)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e828985c-fc85-497a-99de-0176be196f89)

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume x<sub>0</sub><sup>(i)</sup> = 1 for (i belongs to 1, ...., m).
This allows us to do matrix operations with theta and x. Hence making the two vectors 'θ' and x<sup>(i)</sup> match each other element-wise (that is, have the same number of elements: n+1).]

# Gradient Descent for Multiple Variables
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2b83fde3-9b0c-4a2e-a559-1a2cbab94795)

Where X<sub>0</sub> = 1

Gredient descent looks like this:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/19ca596d-5549-472b-a9bf-8b2c215d86ba)

let's see how this partial derivative looks like:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8d285975-0d1e-43e3-b916-2d10837d0407)

Some of the practical tricks to make Gradient Descent work well:
# Gradient Descent in Practice I - Feature Scaling
If you have a problem where you have multiple features, if you make sure the features are on similar scale, then Gradient Descent may converge more quickly.
E.g. x<sub>1</sub> = size (0-2000 feet<sup>2</sup>)
     x<sub>2</sub> = number of bedrooms (1 - 5)
     
If we plot cost function for such function, it will be very long skinny plot.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d921ee48-52ad-4b9f-938d-1150108f065d)

It's going to take much larger time in converging.

In this setting useful thing to do is to scale the features.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f5e1ba24-0942-4069-b3e2-2b187dcf7aeb)

For feature scaling:
- Get every feature into approximately a -1 <= x<sub>i</sub> <= 1 range

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/124ff497-83e7-4c3b-9541-72ea3794a603)

## Mean Normalization
Replace x<sub>i</sub> with x<sub>i</sub> - μ<sub>i</sub> to make features have approximately zero mean (Do not apply to x<sub>0</sub> = 1)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5b61a9de-91bd-4ced-ba69-4f4d8bdf5fbc)

# Gradient Descent in Practice II - Learning Rate
Here is the Gradient Descent update rule:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5e77e310-97e1-44dc-a480-e7a22e162644)

- "Debugging" : How to make sure that Gradient Descent is working correctly.
- How to choose learning rate α

## Making Sure Gradient Descent working Correctly
Job of Gradient Descent is to find the value of θ, which minimizes the cost function J(θ). Plot cost function J(θ) as Gradient Descent runs. If Gradient Descent is working correctly J(θ) should descrease after every iteration of Gradient Descent.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/565f39d3-999b-46e5-9a44-e6263cf135bc)

It can converge after any number of iterations. It's also possible to comeup with automatic convergence test, namely to have an algorithm try to tell you if Gradient Descent has converged.

Example Automatic Convergence Test:
Declare convergence if J(θ) decreases by less than e.g. 10<sup>-3</sup> in one iteration. However, usually choosing this threshold is pretty difficult and so in order to check Gradient Descent's convergance, it's better to look at plots like above rather than relying on automatic convergence test.

Looking at this sort of figure can also tell you, or give you an advance warning, if maybe Gradient Descent is not working correctly.

If J(θ) is increasing with every step of Gradient Descent it mean, Gradient Descent is not working. Most common cause of J(θ) increasing is, if you are trying to minimize a function that may be look like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/73c867ee-421b-4d6f-b3bb-1558971129ea)

And if your learning rate is high Gradient Descent may overshoot the minimum and keep increasing. So the fix is to use smaller value of learning rate α.

Similarly, you might have J(θ) going down for a while and going up for a while and so on. To fix that also we need to use smaller value of learning rate α.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/204e1eca-b94e-4597-90eb-3c2c53bed322)

So bottom line is:
- For sufficiently small α, J(θ) should descrease on every iteration.
- But if α is too small, Gradient Descent can be slow to converge.

Try running Gradient Descent with range of values of α, i.e. 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 etc.. with roughly 3 fold increase and plot J(θ) and then those α for which J(θ) seems to be decreasing rapidly.

# Features and Polynomial Regression
Depending upon the choice of features that you have, you can have different learning algorithms, sometimes very powerful ones by choosing appropriate features.
Ploynomial Regression allows to use machinery of Linear Regression to fit very complicated, even very non-linear functions.

Let's take the example of predicting the price of the house. Suppose you have two features, the frontage of the house and the depth of the house:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/54bc73b3-b483-4da4-9f57-d6007c515621)

You might build the linear regression model like this:
h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub> x frontage + θ<sub>2</sub> x depth

Where:
Frontage is first feature X<sub>1</sub>
Depth is second feature X<sub>2</sub>

But when you are applying linear regression, you don't necessarily have to use just the features X<sub>1</sub> and X<sub>2</sub>X<sub>2</sub> that were given. What you can do is actually create new features by yourself.

We might decide what really determines the price of the house, we might define a new feature Area A.
X = frontage x depth

And then our hypothesis might be:
h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub> x X

Using just one feature, which is land area. So depending upon what insight you might have on a particular problem, rather than just taking the features as it is, sometimes defining the new features might give us better model.

Closely related to the idea of choosing your features is this idea called, Polynomial Regression. 
Let's say you have a housing price data set that looks like this:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/623cb312-d730-48a1-a429-2eb16c558b8d)

Then few different models you might fit to this. One thing you could do is fit a quadratic model like this:
h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub> x X + θ<sub>2</sub> x X<sup>2</sup>

And it might give a fit like below:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c1ffea97-b8e7-430c-bb29-9668c3566b59)

But then you might decide that your quadratic model doesn't make sense because of a quadratic function, this function eventually comes back down and well, we don't think housing prices should go down when the size goes up too high.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e0359825-0814-411b-8e7a-7dadca9e9944)

So then we may be choose a different polynomial model and choose to use instead a cubic function:
h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub> x X + θ<sub>2</sub> x X<sup>2</sup> + θ<sub>3</sub> x X<sup>3</sup>

And we might get this sort of fit:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/26be4762-8c2d-4135-8e85-fd3f26c84eb4)

And green line is somewhat better fit to the data as it doesn't eventually come down.

So how do we actually fit a model like this to our data?
Using the machinery of Multivariate Linear Regression, we can do this with a pretty small modification to our algorithm. 

The form of hypothesis we know how to fit looks like this:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/df25959d-afdd-4441-b2df-761c6f24ef07)

One thing to keep in mind is, if you choose features like this feature scaling becomes increasingly important.
So if the size of house ranges from 1-1000, then (size)<sup>2</sup> ranges from 1-1000,000 and (size)<sup>3</sup> ranges from 1-10<sup>9</sup>

So, these three features take very different ranges of values and feature scaling is absolutely necessary.

Just to give you an example, there might be other reasonable choices. E.g.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/67027245-eca7-40d2-859f-c89de840cc61)



But then you might decide that your quadratic model doesn't make sense because of a quadratic function, this function eventually comes back down and well, we don't think housing prices should go down when the size goes up too high.important.
