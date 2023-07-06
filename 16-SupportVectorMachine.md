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

# SVM: Large Margin Intuition
Here is the cost function for SVM. On the left we have plotted Cost<sub>1</sub>(z) and on the right Cost<sub>0</sub>(z).

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/4ba2410c-e6af-482b-ac2c-69f439a69c5e)

Now let’s think about what it takes to make these cost functions small:
If y = 1 then cost<sub>1</sub>(z) is zero only when z >= 1
If y = 0 then cost<sub>0</sub>(z) is zero only when z <= -1

This is the interesting property of Support Vector Machine, which is that if you have a positive example, so if y = 1, then all we really need is that θ<sup>T</sup>x >= 0 and that would mean that we classify correctly because if θ<sup>T</sup>x >= 0 our hypothesis will predict zero and similarly if you have negative example, what we want is θ<sup>T</sup>x < 0 and that will make sure we got the example right.

But the Support Vector Machine wants a bit more than that. It says, don’t just barely get the example right. So then don’t just have it little bigger than zero. What we really want is for this to be quite a lot bigger than zero. Say may be greater than or equal to 1. And when we want it to be less than 0 we want less than or equal to -1. So this builds in extra safety factor or safety margin factor into the Support Vector Machine.

Logistic Regression does something similar too of course, but let’s see what happens or let’s see consequences of this are in the context of the Support Vector Machine.

Concretely, what we would like to do next is consider a case where we set this constant C (in the cost function) to be a very large value, so let’s imagine we set C to a very large value, may be C = 100,000. Let’s see what Support Vector Machine will do.

# SVM Decision Boundary
If C is very very large, then when minimising this optimisation objective, we are going to be highly motivated to choose a value so that this first term is equal to zero.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9963fcfb-9621-4169-a82a-da4cc65336db)

So let’s try to understand the optimisation problem in the context of, what would it take to make this first term in the objective equal to 0 and this should give us additional intuition about what sort of hypothesis a Support Vector Machine learns.

We saw already whenever we have a training example labeled of y<sup>(i)</sup> = 1, if you want to make that first term zero, what you need is to find a value of θ so that θ<sup>T</sup>x<sup>(i)</sup> >= 1. And similarly whenever we have example with y<sup>(i)</sup> = 0, in order to make sure that the cost<sub>0</sub>(z) is zero we need  θ<sup>T</sup>x<sup>(i)</sup> <= -1.

So if we think of optimisation problem as now, really choosing parameters and ensure that this first term is equal to zero. What we are left with is following optimisation problem:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1320994c-47e1-41c5-b69b-be9c51361b91)

It turns out that when you solve this optimisation problem, when you minimise this as a function of the parameters theta you get a very interesting decision boundary.

# SVM Decision Boundary: Linearly separable case
Concretely, if you look at a data set like below with positive and negative examples :

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2b3b89a7-f5a7-45f1-9c90-dfaae4368d00)

This data is linearly separable, which means there exist many straight lines which can separate positive and negative examples perfectly.

SVM corresponds to the black line in the above pic and that seems like a much better decision boundary than the ones in magenta and green. The black line seems like a more robust separator. Mathematically what that does is, this black decision boundary has a larger distance that distance is called margin, denoted with blue lines. We see that the black decision boundary has some larger minimum distance from any of my training examples whereas the magenta and green lines, they come awfully close to the training examples and that seems to do less good job in separating the positive and negative classes than the black line. This distance is called the margin of the SVM and this gives the SVM certain robustness, because it tries to separate the data with as large a margin as possible. 

So the Support Vector Machine is sometimes also called a Large Margin Classifier and this is actually a consequence of the optimisation problem we wrote earlier.

# Large Margin Classifier in presence of outliers
When C is very large we wrote it as Large Margin classifier. Given a dataset like below, we’ll chose that decision boundary that separates the positive and negative examples on large margin.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6c7cb8e8-5164-41e6-8717-9c69ee1c6862)

SVM is actually slightly more sophisticated than this large margin view might suggest. And in particular if all you are doing is use a large margin classifier then your learning algorithms can be sensitive to outliers.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c302b7c6-3e24-4fb0-8e2a-5ae266461a90)

So let’s just add an extra positive example. If we add one example seems as if to separate the data with large margin may be we endue with decision boundary like below. And it’s really not clear that based on the single outlier based on a single example and it’s really not clear that it’s actually a good idea to change my decision boundary from the black one over to the magenta one.

So if C was very large then this is what SVM will do i.e. change the decision boundary from black to the magenta one but if C was reasonably small then you will still end up with this black decision boundary. 

And of course if the data were not linearly separable so if you had some positive examples in negative side or if you had some negative examples on positive side like below then also SVM will do the right thing and so this picture of large margin classifier that’s really the picture that gives better intuition only for the case of when the regularisation parameter C is very large. C plays role similar to 1/λ

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3210b2a6-5a0c-4e79-b8ed-c4da887a963b)

# SVM - Mathematics Behind Large Margin Classification:
In order to get started, let us first look at couple of properties of what Vector Inner Product looks like.

## Vector Inner Product
Let's say we have two 2D vectors u and v that look like as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e9e2c05b-470b-46ba-af55-b2a03c91ffc4)

Then let's see what u<sup>T</sup>v looks like. And u<sup>T</sup>v is also called the inner product of u and v. Vector u can be represented as:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d72ad51b-51d8-43ce-bdb8-8d3eb180687f)

One quantity that will be nice to have is the norm or euclidean length of vector u represented as ||u||, which is equal to √(u<sub>1</sub><sup>2</sup> + u<sub>2</sub><sup>2</sup>) that's a real number.

Now let's look at vector v, it will be some other vector. Now let's look at how to compute inner product between u and v. Here's how you can do it:
Take the vector v and project it down on vector u and measure the lenght of this red line. Let's call it P.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b30b4e5f-ab01-4050-8003-4155eb96d42b)

P = length of the projection of the vector v on vector u.
u<sup>T</sup>v = P . ||u|| = u<sub>1</sub> v<sub>1</sub> + u<sub>2</sub> v<sub>2</sub> = v<sup>T</sup>u

So, this is one way to compute the inner product. u<sup>T</sup>v is regular multiplication of two real numbers. One more detail: P is actually signed it's positive if andle between u and v is less than 90 degrees else it's negative.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e619e319-f9eb-4bec-9d89-01adc75c0e46)

