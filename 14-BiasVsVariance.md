![image](https://github.com/vivekprm/coursera-ml/assets/2403660/698eff0f-9712-4985-8772-98452fdd4ad9)If you run a learning algorithm and it doesn’t do as well as you are hoping, almost all the time it will be because you have either a higher bias problem or high variance problem. In other words, either an underfitting problem or an overfitting problem. In this case, it’s very important to figure out which of these two problems is bias or variance or a bit of both that you actually have. Because knowing which of these two things is happening would give a very strong indicator for what are the useful and promising ways to try to improve your algorithm.

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

# Learning Curves
Learning Cureves is often very useful thing to plot. If either you wanted to sanity check that your algorithm is working correctly or if you want to improve the performance of the algorithm. It's a tool that people use pretty often to try to diagnose if a physical learning algorithm may be suffering from bias or variance or bit of both.

To plot a learning curve what we usually do is plot J<sub>train</sub> or J<sub>cv</sub> and plot that as a function of m. m is usually a constant like may be we have 100 training examples but what we are going to do is artificially reduce our training set size, So deliberately limit myself to using only let’s say 10 or 20 or 30 training examples and plot what training error is and what the cross validation error is for these smaller training set sizes.

Suppose we have only one training example like that shown In first picture and suppose we are fitting a quadratic function.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c3d34cf4-6259-4333-9f20-0c5f6dd5e1f6)

We will have zero error on one training example. Similarly for two training example we are going to fit it perfectly even if we are using regularisation. If we have three training examples we can fit the quadratic function perfectly. So for m = 1,2,3 J<sub>train</sub>(θ) = 0.
So what we see is if training set size is small what we see is training set error is also small. WIth larger training set it becomes harder and harder to fit it perfectly.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c24e6412-9e6a-4a6f-bffd-7b68ede9e26d)

So training set error tend to increase as size of training set increases. However, cross validation error decreases as size of training set increases. As we have larger training set we might have fitted or model well so cross validation errors should be less.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b197ad97-d046-4a7a-9431-2e017bf36e48)

Let’s look at what Learning curve may look like if we have either higher bias or higher variance problems.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5d037ba7-7d87-4d91-b7da-1bfe3899ec97)

In case of high bias let’s say we fit a straight line to our data. Now if we increase the size of training set we pretty much get the same straight line. In high bias case we find that training error end up getting pretty close to cross validation errors, because you have so few parameters and so much data  the performance on training set and cross validation set will be pretty similar.

And both training and cross validation errors are high in case of high bias. This also implies something very interesting that, If a learning algorithm is high bias as we get more and more training example, we notice that Cross Validation error is not going down much, it’s basically flattened out. So if a learning algorithm is suffering from high bias getting more training data is not going to help.

Next let’s look at a learning algorithm that has high variance

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/81773c0f-22bb-458e-b32c-e4893a730f71)

As we have less training examples we might be fitting our data perfectly. When training set size increases we might not be able to fit the data that perfectly but still pretty close.

So we see large gap between the training error and cross validation error. We can tell from the figure if we keep adding training examples Cross validation error will keep coming down. So in high variance setting getting more training data is indeed likely to help.

So training set error will be pretty low. But what about cross validation error? Cross validation error remain high even if we get moderate size training examples.

# Deciding What to Do Next
## Debugging a learning algorithm
Suppose you have implemented regularised linear regression to predict housing prices. However when you test your hypothesis in new set of houses, you find that it makes unacceptably large errors in its predictions. What should you try next?

We saw that we had many options:
- Get more training examples
- Try smaller set of features
- Try getting additional features
- Try adding polynomial features (x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, x<sub>1</sub>x<sub>2</sub>, etc.)
- Try decreasing λ
- Try increasing λ

Is there a way to figure out which of these might be fruitful options? Below are some indications

- Get more training examples - Good for fixing high variance
- Try smaller set of features - Good for fixing high variance
- Try getting additional features - Usually but not always fixes high bias problem
- Try adding polynomial features (x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, x<sub>1</sub>x<sub>2</sub>, etc.) - Fixes high bias problem
- Try decreasing λ - fixes high bias
- Try increasing λ - fixes high variance

Finally let’s take everything that we have learned and relate to Neural Network and see some practical advice to usually choose the architecture and connectivity pattern of the Neural Network

# Neural Networks and Overfitting
If you are fitting Neural Network, one option would be to fit, say a pretty small Neural Network, relatively very few hidden units. So network like this might have relatively few parameters and be more prone to underfitting. The main advantage of these small Neural Networks is that the computation will be cheaper.

Alternative will be to fit a very large Neural Network with either more hidden units or with more hidden layers and so these Neural Networks tend to have more parameters and therefore be more prone to overfitting. One disadvantage but often not a major one but something to think about, is that if you have a large number of Neurons in your Neural Network then it can be more computationally expensive. 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8c458597-41c7-4434-95ed-c5de7c8b3f1c)

The main potential problem of these much larger Neural Networks is that it could be more prone to overfitting. And it tuns out if you are applying Neural Network very often using a larger Neural Network often the larger the better. But if it’s overfitting then use regularisation to address overfitting and usually using a larger Neural Network by using Regularisation to address overfitting that’s often more effective than using a smaller Neural Network.

And finally, one of the other decisions is, the number of hidden layers you want to have? Do you want one hidden layer or do you want two hidden layer or three hidden layers etc?

Using a single hidden layer is reasonable default but if you want to choose number of hidden layers, one other thing you can try is find yourself a training, cross validation and test set split and try training Neural Networks with one hidden layer, two hidden layers, three hidden layers and so on and see which of those Neural Networks performs best on the cross validation set.
