If you run a learning algorithm and it doesn’t do as well as you are hoping, almost all the time it will be because you have either a higher bias problem or high variance problem. In other words, either an underfitting problem or an overfitting problem. In this case, it’s very important to figure out which of these two problems is bias or variance or a bit of both that you actually have. Because knowing which of these two things is happening would give a very strong indicator for what are the useful and promising ways to try to improve your algorithm.

# Bias/Variance

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f687a47f-3074-4d5c-b10a-640c267610ed)

Below are our training and cross validation errors:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c61b4ac9-3f28-4dc2-9e12-51f38557ae80)

This sort of plot also helps us to better understand the notions of bias and variance.

Concretely, suppose that you have applied a learning algorithm and it’s not performing as well as you are hoping. So if your cross validation set error J<sub>cv</sub>(θ) or your test set error J<sub>test</sub>(θ) is high, how can we figure out if the learning algorithm is suffering from high bias or high variance?

So, the setting of cross-validation error being high belongs to the region with blue circle and left circle corresponds to high bias problem whereas right circle corresponds to high variance problem..

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/62c5bb8b-99f7-4968-bf3d-4559c6b6d19f)

# Regularisation and Bias/Variance
We’ve seen how regularisation can help prevent over-fitting but how does it affect the bias and variances of a learning algorithm?

## Linear Regression with regularization
Suppose we are fitting a high order polynomial, (like shown below) but to prevent overfitting we need to use regularisation (as shown below)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3906169a-df01-49cc-8ab0-80e2eca2cafd)

Let's consider three cases:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b42ce76a-46f2-4f31-849b-2300fd0e9f1c)

So how can we automatically choose a good value for the regularization parameter?
Just to reiterate below is our model and our learning algorithms objective:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/26c8c7e2-ba60-464b-984a-8d8340f97b6e)

For the settings where we are using regularization, lets define J<sub>train</sub>(θ). When we were not using regularization we defined J<sub>train</sub>(θ) to be the same as J(θ) but when we are using regularization term we are going to define J<sub>train</sub>(θ) to be sum of squared errors on the training set without taking into account that regularisation. Similary we are also going to define Cross Validation set and test set error.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ff02d130-3dd4-4444-9e0c-e4e3359e1049)

So this is how we can automatically choose the regularisation.

## Choosing the regularisation parameter λ
May be have some range of values of λ that we want to try out. E.g.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/af82759f-be8e-40a5-b6c1-e499003a59e1)

So this gives me may be 12 different models that we are trying to select amongst corresponding to 12 different values of λ. Of course you can go to values less than 0.01 or values larger than 10. Given each of these 12 models what we can do is the following:
- Take the first model for λ = 0, Minimize cost function J(θ) and it will give us some parameter vector θ<sup>(1)</sup>
- Similarly take the second model and calculate θ<sup>(2)</sup> and so on.
- Take all of these hypothesis and parameter vector theta and use Cross Validation set to validate them.
- Choose the model with lowest cross validation set error. And let's say for the sake of this example we ended up picking θ<sup>(5)</sup>.
- Report test set error  J<sub>test</sub>(θ<sup>(5)</sup>)

Last thing that we want to understand is how cross validation and training error vary as we vary the regularization parameter λ

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/640dbaf4-e0ba-40ff-bc67-333d0a409fa6)

For small values of λ regularisation term goes away. So when λ is small you end up with small value of J<sub>train</sub>(θ). Wereas if lambda is large then you havve high bias problem and you might not fit your training set well. So J<sub>train</sub>(θ) increases with increase in λ.

If we have large value of λ we may endup underfitting so we may have high bias problem and so cross validation errors will be high. However, if we have smaller value of λ we may be overfitting the data and so again cross validation errors will be high.

And so it will be some intermediate value of λ that works best in terms of having small cross validation error and test error.
