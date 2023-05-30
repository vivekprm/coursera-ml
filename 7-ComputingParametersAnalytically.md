# Normal Equation
Gradient Descent is an Iterative Algorithm that takes many steps, multiple iterations of Gradient Descent to converge to the Global Minimum. In contrast Normal Equation 
gives us a method to solve for θ analytically. So, rather than needing to run this iterative algorithm, we can instead just solve for the optimal value for θ all in one
go. So that basically in one step you get to the Optimal Value.

It turns out that Normal Equation method has some advantages and some disadvantages. But before we talk about that let's get some intuition about what this method does.
Let's take very simplified cost function:
J(θ) = aθ<sup>2</sup> + bθ + c where θ belongs to R

So imagine θ is just a scalar value or real value. It's just a number rather than a vector. Well, how do we minimize the quadratic function?

If you know the Calculus, the way to minimize a function is to take derivatives and set derivatives equal to 0. So,

d/dθ J(θ) = 0

This allows to solve for θ that minimizes J(θ). That was the simple case when θ was just real number. In the problem that we are interested in, θ is no longer just a 
real number but instead is this n+1 dimensional parameter vector and a cost function J is a function of this vector.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/147ab498-babc-4ad4-ad3b-6891fb2ba3b0)

How do we minimize this cost function J. Calculus tells us, one way to do so is, to take the partial derivative of J, with respect to every parameter of θ<sub>j</sub>
in turn and then set all of these to 0 and solve for θ<sub>0</sub>, θ<sub>1</sub>, ..., θ<sub>n</sub>. Then this will give the value of θ to minimize the cost function.

Example: m=4

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0c55c44c-814b-4a87-a13b-d7accc55b58c)

Let's say we have 4 training examples. In order to implement this normal equation solution, we are going to take our data set of 4 training examples. What we are going
to do is add an extra column that corresponds to my extra feature X<sub>0</sub>. 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f0eef521-7f6b-4fe8-826a-a670995af971)

Then we are going to construct a matrix called X that is:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2d857f57-837f-4543-9a31-6464765565a6)

Which basically contains all our feeatures from the training data.

And now take values that we are going to predict and construct a vector y like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c4371aa4-135f-4fd5-88f9-bf5cfaefabb1)

And so X is going to be m x (n+1) dimensional matrix and Y is going to be m dimensional vector.

Finally if you compute below equation:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0a8006b6-6516-450e-b46c-41d8069a8544)

This will give us the value of θ that minimizes the cost function.

In general case let's say we have m training examples (x<sup>(1)</sup>, y<sup>(1)</sup>), ...., (x<sup>(m)</sup>, y<sup>(m)</sup>); n features

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1bd0e797-da3f-4585-a144-0c8e50ad3d92)

Octave: pinv(X' * X) * X' * Y

If we are using Normal Equation, fature scaling is not really necessary.

Finally whn should you use Gradient Descent and when to use Normal Equation?

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/cd796f05-8271-4811-af57-2350f5f5ddb5)

If n is of the order of 100, inverting 1000 x 1000 matrix is no problem by modern computing standard.

It's hard to give a strict number for which prefer Gradient Descent over Normal Equation. 

# Normal Equation Noninvertibility
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/cc86d19b-fb64-434a-b053-85020c1e3bb0)

What if X<sup>T</sup>X is non-invertible? (singular/degenerate). It should happen pretty rarely. Octave does the right thing. Octave has two functions for inverting
matrices one is called pinv and other is called inv. The differences between these two is somewhat technical. One is called pseudo inverse and other ine is called
inverse. But you can show mathematically that, so long you use pinv function then this will actually compute the value of θ that you want even if X<sup>T</sup>X is 
non-invertible.

There are mainly two most common causes why X<sup>T</sup>X is non-invertible:
- Redundant features (linearly dependent)
  -  E.g. x<sub>1</sub> = size in feet<sup>2</sup>
  -  x<sub>2</sub> = size in m<sup>2</sup>
- Too many features (e.g. m <= n)
  - Delete some features or use regularization (which allows to fit lots of features even if we have small number of training data)

However, this should not be a problem with most implementations of Linear Regression.
