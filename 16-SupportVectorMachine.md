Compared to Logistic Regression and Neural Networks, Support Vector Machine sometimes give cleaner and more powerful way of learning complex non-linear functions.
In order to describe Support Vector Machine, we are actually going to start with Logistic Regression and show how we can modify it a bit, and get what is essentially Support Vector Machine or SVM.

Logistic Regression Hypothesis & Sigmoid Activation Function(right):

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f8b26dc2-707a-4703-8db4-61bc15916ad9)

Now let’s think about what we would like Logistic Regression to do. If we have an example with y = 1, which means an example either in the training set or the test set or the cross validation set. But when y=1 then we are sort of hoping that h<sub>θ</sub>(x) will be close to one, so we are hoping to correctly classify the problem.

h<sub>θ</sub>(x) ~ 1 means θ<sup>T</sup>x must be larger than 0. From the graph we can see when z is much bigger than 0 i.e. far right in the picture then the output of Logistic Regression becomes close to 1.

Conversely, if we have an example where y = 0, then what we are hoping for is that the hypothesis will output a value close to 0 and that corresponds to θ<sup>T</sup>x or z being much less than 0 because that corresponds to a hypothesis of putting a value close to 0.

If you look at the cost function of Logistic Regression what you will find is that each example (x, y) contributes a term like this to overall cost function.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/278cc0a9-1c65-4845-84bd-665931db6dad)

So for the overall cost function, we will have a sum over all the training examples I.e. 1 to m, but the term here is what a single training example contributes to the overall objective function. 

Now let’s consider two cases of when y=1 and y=0:

When y=1 the second term becomes 0 and we get plot like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/83107834-063a-4440-922f-8c518165026f)

We see that when z is large I.e. θ<sup>T</sup>x is large that corresponds to a value of z that gives us a very small value, a very small contribution to the cost function. And this explains why, when logistic regression sees a positive example with y=1, it tries to set θ<sup>T</sup>x to be very large.

Now to build the Support Vector Machine, here is what we are going to do, we’re going to take this Cost Function and modify it a little bit.

Let’s take the point 1 on z axis, the new Cost Function can be a Flat line from here and grows as Straight Line (as in below figure) similar to Logistic Regression.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b53a4fc2-4cdb-46f8-8d7a-df17fc82865e)

So the curve that we draw here in magenta colour, is pretty close approximation to the Cost Function used by Logistic Regression. Except it’s now made up of two line segments, there is this flat portion on the right and then there’s this straight line portion on the left.

That’s the new cost function that we are going to use when y=1 and you can imagine it should do something pretty similar to Logistic Regression. But turns out that this will give the Support Vector Machine computational advantages and give us, later on, an easier optimization problem.

The other case is y=0, in that case first term will be equal to zero in above equation and Cost Function will be equal to the Second Term. If you plot that as function z, it looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/7f01479c-901a-4f62-a885-e830ed3d7722)

For Support Vector Machine again we are going to replace the Blue Line with something similar. Let’s call this Cost Function as Cost<sub>0</sub>(z).

# Support Vector Machine
Here is the Cost Function J(θ) that we have for Logistic Regression:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/865e377b-85e1-4f19-9a85-05b716afcc7e)

For the Support Vector Machine what we are going to do is take logarithm terms in bracket and replace with Cost<sub>1</sub>(θ<sup>T</sup>x) and Cost<sub>0</sub>(θ<sup>T</sup>x)

So what we have for Support Vector Machine is:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/67bdf3e5-99c2-41c0-9474-5f04cec08601)

Now by convention for Support Vector Machine, we actually write things slightly different. We re-parameterise this just very slightly differently. 
- First we are going to get rid of 1/m term, as that doesn’t change the min.
- For Logistic Regression, we had two terms to the objective function:
    - First is the Cost that comes from training set
    - Second is the regularisation term.
    - Which can be represented as A + λ B, and we were controlling the tradeoff between A and B using this λ. By setting different values for this regularisation parameter λ we could tradeoff the relative weight between how much we want to fit the training set well versus how much we care about keeping the values of the parameters small.
    - For SVM just by convention, we are going to use a different parameter. So instead of using  λ here to control the relative weighting between first and second term, we are instead going to use a different parameter which by convention is called C and instead we are going to minimise CA+B. So for Logistic Regression if we set a very large value of λ that means you will give B a very high weight. In this case if we set C to be a very small value then that corresponds to giving B a much larger rate than C, than A. So this is just different way of controlling the tradeoff.

So the cost function for SVM looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/87a74643-037f-45af-abdb-15e6569504e5)

# SVM Hypothesis

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/85160207-df11-497f-97de-c93a2c2ab0db)
