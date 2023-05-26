Our first learning algorithm will be **Linear Regression**.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/989182a1-eac2-4353-82d3-e96f648e5691)

It's a Linear Regression Problem. We need a set of training set data to make predictions:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/948bfa85-56d3-4dd6-abd0-fda3b3328711)

To establish notation for future use, we’ll use x<sup>(i)</sup> to denote the "input" variables (living area in this example), also called input features, 
and y<sup>(i)</sup> to denote the "output" or target variable that we are trying to predict (price).

A pair (x<sup>(i)</sup>, y<sup>(i)</sup>) is called a training example, and the dataset that we’ll be using to learn—a list of **m training examples**(x<sup>(i)</sup>,y<sup>(i)</sup>); i=1,…,m — is called a training set.

Note that the superscript "(i)" in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, X = Y = ℝ.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a "good" predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5d96b2b6-14a7-43eb-92e7-9c7b035becfa)

When the target variable that we’re trying to predict is **continuous**, such as in our housing example, **we call the learning problem a regression problem**. When y can take on only a **small number of discrete values** (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a **classification problem**.

# Cost Function
We'll define something called the cost function, **this will let us figure out how to fit the best possible straight line** to our data.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/106b67fe-ec9f-4a2b-8fa7-843f7242b546)

Our straight line looks as below for different values of 𝛉<sub>0</sub>, 𝛉<sub>1</sub>:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8eb73687-84fa-4fed-90cc-95445e4e2744)

Idea: Choose 𝛉<sub>0</sub>, 𝛉<sub>1</sub> such that h<sub>𝛉</sub>(x) is close to y for our training examples (x, y):

Minimize (h<sub>𝛉</sub>(x) -y)<sup>2</sup> for 𝛉<sub>0</sub>, 𝛉<sub>1</sub>. For the whole training sets.

Σ (h<sub>𝛉</sub>(x) - y)<sup>2</sup>. Should be minimised. Or, 1/2m Σ (h<sub>𝛉</sub>(x) - y)<sup>2</sup> should be minimum. Which makes math little easier.

Which is the cost function:
                   
J(𝛉<sub>0</sub>,  𝛉<sub>1</sub>) = 1/2m Σ (h<sub>𝛉</sub>(x<sup>i</sup>) - y<sup>i</sup>)<sup>2</sup> , where 1 <= i <= m
                                      
Minimize J(𝛉<sub>0</sub>, 𝛉<sub>1</sub>). Cost function is also called squared error function.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/67766bbf-e67a-4901-92ba-40a26696c947)

# Cost Function Institution 1
If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by h<sub>𝛉</sub>(x)) which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of J(θ<sub>0</sub>, θ<sub>1</sub>) will be 0. The following example shows the ideal situation where we have a cost function of 0.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1e6310aa-3eb8-44d3-a0f5-005f4763f59b)

When θ<sub>1</sub> = 1, we get a slope of 1 which goes through every single data point in our model. Conversely, when θ<sub>1</sub> = 0.5, we see the vertical distance from our fit to the data points increase.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/59124564-18b8-4adb-a9fb-95dc7066572a)

This increases our cost function to 0.58. Plotting several other points yields to the following graph:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e5c1d07f-e580-4218-9aba-2aba8a9c9d95)

Thus as a goal, we should try to minimize the cost function. In this case, θ<sub>1</sub> = 1 is our global minimum.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a7c72555-5853-488c-b208-9946b00d1478)

## Cost Function - Intuition II
A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example 
of such a graph is the one to the right below.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8b8a1714-6016-4d2e-88b2-ea5c7d8d7da9)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for J(θ<sub>0</sub>, θ<sub>1</sub>) and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when θ<sub>0</sub> = 800 and θ<sub>1</sub> = -0.15. Taking another h(x) and plotting its contour plot, one gets the following graphs:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6c3d9896-76cc-4be4-861d-7d32110ebedf)

When θ<sub>0</sub> = 360 and θ<sub>1</sub> = 0, the value of J(θ<sub>0</sub>, θ<sub>1</sub>) in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6c1d88c9-4324-4703-aa05-345b585699a5)

The graph above minimizes the cost function as much as possible and consequently, the result of θ<sub>1</sub> and θ<sub>0</sub> tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

# Gradient Descent
So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields θ<sub>0</sub> and θ<sub>1</sub> (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put θ<sub>0</sub> on the x axis and θ<sub>1</sub> on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/84abcfbf-29b6-4994-8cdc-0dd3aea4a113)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. **The slope of the tangent is the derivative at that point and it will give us a direction to move towards**. We make steps down the cost function** in the direction with the steepest descent**. The size of each step is determined by the parameter α, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of J(θ<sub>0</sub>, θ<sub>1</sub>). Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

repeat until convergence:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0376a4d9-793a-4cc8-8a7b-851f6539d3d6)

where
j=0,1 represents the feature index number.

At each iteration j, one should simultaneously update the parameters θ<sub>1</sub>, θ<sub>2</sub>,..., θ<sub>n</sub>. Updating a specific parameter prior to 
calculating another one on the j(th) iteration would yield to a wrong implementation.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/4cb024f0-88c7-471a-8edf-1cdac952f982)

# Gradient Descent Intuition
We explored the scenario where we used one parameter *θ*1 and plotted its cost function to implement a gradient descent. Our formula for a single parameter was :

Repeat until convergence:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/67068232-6560-44d2-8ddf-33e26def4fc8)

Regardless of the slope's sign for d/dθ<sub>1</sub>J(θ<sub>1</sub>), θ<sub>1</sub> eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of θ<sub>1</sub> increases and when it is positive, the value of θ<sub>1</sub> decreases.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e0194406-f8de-48cd-929a-7e7e48861d6d)

On a side note, we should adjust our parameter α to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b818f212-c5ff-4ea2-afb2-6d96c1a60105)


## How does gradient descent converge with a fixed step size α?
The intuition behind the convergence is that d/dθ<sub>1</sub>J(θ<sub>1</sub>) approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:

θ<sub>1</sub> := θ<sub>1</sub> − α * 0

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/7937fa6b-14db-47cf-af70-eaf16f4cc42d)

# Gradient Descent For Linear Regression
When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/138d6638-13b6-4c8d-b78c-77d90d933bb6)

where m is the size of the training set, θ<sub>0</sub> a constant that will be changing simultaneously with θ<sub>1</sub> and x<sub>i</sub>, y<sub>i</sub> are values of the given training set (data).

Note that we have separated out the two cases for θ<sub>j</sub> into separate equations for θ<sub>0</sub> and θ<sub>1</sub>; and that for θ<sub>1</sub> we are multiplying x<sub>i</sub> at the end due to the derivative. The following is a derivation of ∂/∂θ<sub>j</sub>J(θ) for a single example :

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/784233bd-0df2-458c-bda6-8697aefdb536)

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus **gradient descent always converges (assuming the learning rate α is not too large) to the global minimum**. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/01457df3-3539-48a5-8bde-d804857826d1)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.
