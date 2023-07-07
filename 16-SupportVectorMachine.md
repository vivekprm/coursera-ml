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

So that's how vector innver product works. We are going to use these properties of vector inner product to try to understand the Support Vector Machine optimisation objective.

## SVM Decision Boundary
Here is the optimisation objective for SVM.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/94c8a2d6-0ac9-49a1-bfb0-cacbfe91ebe5)

We have made one simplification just to make objective easy to analyze. What we are going to do is ignore θ<sub>0</sub> i.e. θ<sub>0</sub> = 0.

To make things easy to plot we are also going to set n the number of features to be equal to 2. So we have only two features. Our optimization objective now becomes 1/2(θ<sub>1</sub><sup>2</sup> + θ<sub>2</sub><sup>2</sup>) which can be rewritten as 1/2(√(θ<sub>1</sub><sup>2</sup> + θ<sub>2</sub><sup>2</sup>)<sup>2</sup>) which is equal to 1/2(||θ||)<sup>2</sup>

So all SVM is doing in the optimisation objective is it's minimizing the squared norm of parameter vector θ.

Now let's look at θ<sup>T</sup>x<sup>(i)</sup> and understand better what they are doing. So given the parameter vector θ and given example x<sup>(i)</sup> what is θ<sup>T</sup>x<sup>(i)</sup> equal to?

Considering the Vector inner product, let's say we have single positive example x<sup>(i)</sup> as below: 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b2e0c7a2-477c-485d-b862-df8f7f9f6b58)

So, θ<sup>T</sup>x<sup>(i)</sup> can be replaced with P<sup>(i)</sup>. ||θ||

So wrtiting that down in our optimization objectiva becomes as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/76b12593-f150-4716-9590-6c29bd727b81)

Now let's consider the training example we have below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b6dc54e6-8647-484a-abaf-992acddd64e0)

Let's see what decision boundary SVM will choose, here is one option.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/79a72ff0-9395-4a3d-9a24-9f44853b4db4)

This is not a very good choice because it has very small margins. This decision boundary comes very close to the training examples. Let's see why support vector mahine will not choose this. 

For this choice of parameters it's possible to show that parameter vector θ is actually at 90 degree to the decision boundary. So that green decision boundary corresponds to a parameter vector θ shown with blue line.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ad4f2508-e0d4-48a4-a1d0-9431cfbafd11)

The simplification θ<sub>0</sub> = 0, means that the decision boundary has to pass through origin. Now let's look at what this implies for the optimisation objective. Let's take the positive example below the x axis, let's consider that as our first example. If we consider the projection of this example over parameter vector θ, we get that little red segment P<sup>(1)</sup> and that's going to be pretty small. 

Similarly if we take x<sub>2</sub> from the negative example it has projection shown with magenta line and it's going to be P<sup>(2)</sup> and it will be a negative number. So what we are finding is these terms P<sup>(i)</sup> are going to be pretty small numbers. For optimization objective we see that P<sup>(i)</sup>. ||θ|| >= 1, that means we need ||θ|| to be very large.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ce1bf6b5-d3e3-44e7-84a8-38acecaa0e8d)

And similarly for negative example P<sup>(2)</sup> . ||θ|| to be <= -1. and we already saw that P<sup>(2)</sup> is going to be pretty small negative number and so only way for P<sup>(2)</sup> . ||θ|| <= -1 to happen ||θ|| has to be large. But what we are doing in the optimisation objective is we are trying to find a setting of parameters where ||θ|| is small, which contradict so this decision boundary doesn't seem in good direction for parameter vector θ.

In contrast, let's look at different decision boundary. Let's say SVM chooses the decision boundary in below pic and we have corresponding parameter vector theta and projections along it:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/21253aa8-3706-4f73-9866-d8f20cc246c1)

Now we see that projections P<sup>(1)</sup> and P<sup>(2)</sup> are bigger now, so ||θ|| can be smaller. Which is why SVM will choose this decision boundary. This is how SVM gives rise to large margin classification effect.

Question: The SVM optimization problem we used is:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c1ad9e32-27ea-49ac-9871-30945f25d135)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/dd6b5e5f-0c52-4247-8f28-567660857b33)

where p<sup>(i)</sup> is the (signed - positive or negative) projection of x<sup>(i)</sup> onto θ. Consider the training set above. At the optimal value of 
θ, what is ||θ||?

Ans: 1/2

# SVM: Kernels I
The main technique to develop complex non linear classifier using SVM is something called Kernel.

## Non-linear Decision Boundary
If you have a training set that looks like this, you want to find a non-linear decision boundary to distinguish the positive and negative examples, may be a decision boundary that looks like in below pic.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/504906a5-e98c-4a93-96c3-45d0b909870d)

One way to do this is to come up with a set of complex polynomial feature like in above pic. So you end up with hypothesis like:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/10de4c4c-9b9a-4142-ae26-4aee2a9d737a)

Another way of writing this, to introduce a little bit of new notation that we will use later, is that we can think of a hypothesis that’s computing a decision boundary using θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + … Where we are going to use this new notation f<sub>1</sub>, f<sub>2</sub>, f<sub>3</sub> and so on to denote these new sort of features that we are computing.

f<sub>1</sub> = x<sub>1</sub>
f<sub>2</sub>= x<sub>2</sub>
f<sub>3</sub> = x<sub>1</sub>x<sub>2</sub>
f<sub>4</sub> = x<sub>1</sub><sup>2</sup>
f<sub>5</sub> = x<sub>2</sub><sup>2</sup>
And so on..

We have seen this previously that coming up with these high order polynomials is one way to comeup with lots more features. But the question is, is there a different choice of features or is there better choice of features than these high order polynomials because you know it’s not clear that this high order polynomial is what we want and when we talked about computer vision we talked about when the input is an image with lots of pixels. We also saw how using high order polynomials becomes very computationally expensive because there is a lot of these higher order polynomial terms.

So is there a different or better choice of features that we can use to plug into this sort of hypothesis form. So here is one idea for how to define new features f1, f2, f3.

On this line we are going to define only three new features, but for real problems we get to define a much larger number.

## Kernel
In this case  we have features x1 and x2 and we are going to leave x0 out of this, we are going to manually pick few points l<sup>(1)</sup>,  l<sup>(2)</sup> and  l<sup>(3)</sup> and for now let’s say that we are going to choose these three points manually.

Now what we are going to do is define my new features as follows:

Given an example X, let me define my first feature:

f1 = similarity(x, l<sup>(1)</sup>) 

We are going to have a formula to measure similarity.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5c9ae73c-d1ad-4fa1-b3d4-65eea251dbd6)

||x - l<sup>(1)</sup>||<sup>2</sup> is the euclidean distance between the point x and the landmark l<sub>1</sub>

Similarly we can compute f2 and f3.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/26be95be-7624-4006-95aa-afd0d9dfafd2)

And what this similarity function is, the mathematical term for this, is that this is going to be a kernel function. And the specific kernel that I am using here, is actually called a Gaussian Kernel.

So this exponential formula, this particular choice of similarity function is called Gaussian Kernel. But the way terminology goes is that in abstract these different similarity functions are called Kernels and we can have different similarity functions adn the specific example we have here is called Guassian Kernel.

Ans so instead of writing similarity between x and l, sometimes we also write this a kernel denoted as k(x, l<sup>(i)</sup>).

So let's see what these kernels actually do and why these sort of similarity functions might make sense. 

## Kernels & Similarity
Let's take our first landmark l<sup>(1)</sup>, which is one of those points that we choose on our figure.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1d7e97f2-5dee-4373-bfa4-d085905f8306)

Just to make sure we are on the same page about what the numerator terms is, the numerator can also be written as a sum from j=1 to n of the distance (component-wise distance between the vector x and l. Here are are ignoring x<sub>0</sub>

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5477e350-a888-4ebb-9e07-82065cfb7989)

So, this is how you compute the Kernel with similarity between X and a landmark. So let's see what this function does.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/58a70f5e-9cf6-4ebc-8359-09911b7890a3)

So what these features do they measure how similar X is from one of your landmarks. And the feature X is going to be close to 1 when X is close to landmark and it's going to be 0 when featue X is far from the landmark.

Each of these landmarks l<sup>(1)</sup>, l<sup>(2)</sup> & l<sup>(3)</sup>, defines a new feature f<sub>1</sub>, f<sub>2</sub> & f<sub>3</sub>.

Given a training example X we can now compute three new features f<sub>1</sub>, f<sub>2</sub> & f<sub>3</sub> given the three landmarks l<sup>(1)</sup>, l<sup>(2)</sup> & l<sup>(3)</sup>.

Let's plot this similarity function and see what it looks like.
For this example let's say we have two features X1 and X2 and lt's say our first landmark l1 is at below location:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/64a7e8ce-32a3-44ec-8d74-c0eeb68183d0)

The height of the surface is the value of f<sub>1</sub>. 

Now let's look at the effect of varying σ<sup>2</sup>. It's called Guassian Kernel parameter and as you vary it you get slightly different effect. Lets set σ<sup>2</sup> to 0.5 and see what we get. We see that curve looks similar except that the bump becomes narrower.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1f344668-eae4-461e-9bac-e101cd1c5aa1)

As you start from l<sup>(1)</sup> i.e. (3, 5) and move away then the feature f1 falls to 0 more rapidly. And conversely If you were to increase σ<sup>2</sup> = 3 in that case as we move away from l<sup>(1)</sup> f1 falls to zero much more slowly.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a9313c59-ac0c-4628-981d-017984a0479f)

So given this definition of the features let's see what sort of hypothesis we can learn.

Given the training example X we are going to compute these features f1, f2, f3 and hypothesis is going to predict 1 when 

θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub> >= 0

For this particular example, let's say we already found a learning algorithm and let's say that somehow we ended up with these values of the parameter:
θ<sub>0</sub> = -0.5, θ<sub>1</sub> = 1, θ<sub>2</sub> = 1, θ<sub>3</sub> = 0

What we want to do is consider what happems if we have a training example that has a location at the magenta dot in below pic. What our hypothesis will predict?

If we look at the forumula θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub> >= 0 and because our point is close to l<sup>(1)</sup> f1 ~ 1 and because it's far from l2 and l3 f2, f3 ~ 0

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e0265e96-067d-462d-b8ed-762613cadea0)

Now let's take a different point denote in cyan if that was my training example X then f1, f2, f3 ~ 0 so it will be equal to -0.5 so we predict y = 0.

So we can see that points close to l1 and l2 we end up predicting y = 1 and for points for away from l1 and l2 we predict y = 0. SO the decision boundary of this hypothesis will look something like below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ec2a53a6-63a5-48c8-9b81-c7800a5c7520)

# SVM: Kernels II
## Choosing the Landmarks

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3aa2da5a-1e32-4e8d-8754-5d7948c6ff4e)

How to choose these landmarks l<sup>(1)</sup>, l<sup>(2)</sup> & l<sup>(3)</sup>? 

For complex learning problems, may be we want lot more landmarks than just three of them that we might choose by hand.

In practice this is how the landmarks are chosen, which is that given a Machine Learning problem, we have some data set of some positive and negative examples, for every training example that we have, we are going to put landmark at exactly the same locations as the training examples. If we have first training example X<sup>(1)</sup> we are going to choose first landmark l<sup>(1)</sup> at exactly the same location as our first training example. Similarly choose other landmarks as in below pic:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1a8284a7-855f-416d-ba2f-b9493b8f9f9b)

It is saying that my features are basically going to measure how close an example is to one of the things that we saw in our training set.

Just to write it down more concretely.

## SVM with Kernels

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e0d7aff3-5830-4780-aa87-7a9293f57e08)

When you are given an example x, and this example x can be something in the training set, something in the cross-validation set or something in the test set, we are going to compute these features f1, f2, .. and so on:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a3896940-8da0-4a73-9a7f-2856b3a01fe0)

In general: 
f<sub>i</sub><sup>(i)</sup> = sim(x<sup>(i)</sup>, l<sup>(i)</sup>) where l<sup>(i)</sup> is equal to x<sup>(i)</sup>. 

So f<sub>i</sub><sup>(i)</sup> is going to be similarity between x<sup>(i)</sup> and itself. IF we are using the Guassian Kernel it is equal to exp(-0/(2σ*σ)) = 1

So, given these kernels and similarity functions, here is how we use Support Vector Machine. If you already have a learned set of parameters θ, then if you are given a value of x and you want to make a prediction, what we do is we compute the features f which is R<sup>m+1</sup> dimensional feature vector. What we do is then, we predict y = 1 if θ<sup>T</sup> f >= 0  which is same as θ<sub>0</sub> f<sub>0</sub> + θ<sub>1</sub> f<sub>1</sub> + ... + θ<sub>m</sub> f<sub>m</sub> >= 0

If you already have a setting for parameters theta. How do you get the parameters theta? 
Well we do that using the SVM learning algorithm and specifically what you do is, you would solve thie minimization problem 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d647b0dd-58c7-4e1b-ae06-08e7c999f42c)

However, instead of using θ<sup>T</sup> x<sup>(i)</sup> we are using new feature θ<sup>T</sup> f<sup>(i)</sup> to make a prediction on ith training example.
By solving this minimization problem you get the parameters for Support Vector Machine.
