Back propagation as an algorithm has lot of details and can be little bit tricky to implement and one unfortunate property is that there are many ways to have subtle bugs in back propagation. So that if you run it with gradient descent or some other optimizational algorithm, it could actually look like it's working and your cost function J(θ) may endup decreasing on every iteration of gradient descent. But this could prove true even though there might be some bug in your implementation of back propagation. So that it looks J(θ) is decreasing but you might just windup with a Neural Network that has a higher level of error than you would with a bug free implementation and you might just not know that there was this subtle bug that was giving you worse performance.

So, what can we do about this?
There is an idea called Gradient Checking that eliminates almost all of these problems. If you do this it will help you make sure and sort of gain high confidence that your implementation of forward propagation and backward propagation or whatever is 100% correct.

# Numerical Estimation of Gradients
Consider the following example, suppose that we have a function J(θ) and have some value θ (assume θ is a real number) and let's say we want to estimate the derivative of this function at this point (see pic below). 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/555c7c88-32b7-4ca0-90e1-d39db0a16a00)

Here is the procedure to numerically approximating the derivative, we are going to compute θ + ε and θ - ε and we are going to look at those two points, connect them by straight line and slope of that line is going to approximation of our derivative.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/07a20bb9-acd3-4afa-b3d2-c2c9926168ac)

The formula on the right (in above pic) is called the one-sided difference, whereas formula on the left is called two sided difference. Two sided difference gives us slightly more accurate estimate. So we use that rather than one-sided difference.

## Implementation in Octave
```
gradApprox = (J(theta + EPSILON) - J(theta - EPSILON))/ (2 * EPSILON)
```

Now let's look at the more general case of when theta instead of being a real number is a Vector Parameter, so let's say theta is an R<sup>n</sup> an unrolled version of the parameters of our Neural Network. So theta is a vector that has n elements.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2c4b5dd5-bbfd-402d-b711-916a6dfb701f)

We can then use the similar idea to approximate all the partial derivative terms.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/add2b3bb-526f-41dc-bfb7-f555fc783650)

Concretely, what we implement is the following:

```
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) = thetaPlus(i) + EPSILON;
  thetaMinus = theta;
  thetaMinus(i) = thetaMinus(i) - EPSILON;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus)) / (2 * EPSILON);
end;
```

The way we use it in our Neural Network implementation is, we would implement this for loop to compute the top partial derivative of the Cost Function with respect to every parameter in the Network and we can then take the gradient that we got from back propagation i.e. DVec

And then take our numerically computed derivative that is this gradApprox and make sure that is approximately equal to the DVec that we got from back propagation.

If it's then we are confident that our implementation of back propagation is correct. And when we plug this DVec vector into Gradient Descent or some Advanced Optimization Algorithm we can then be much more confident that we are computing the derivatives correctly and therefore our code will run correctly and do a good job optimizing J(θ).

## Implementation Note
- Implement backprop to compute DVec (Unrolled D<sup>(1)</sup>, D<sup>(2)</sup>, D<sup>(3)</sup>)
- Implement numerical gradient check to compute gradApprox.
- Make sure they give similar values.
- Turn off Gradient checking. Using backprop code for learning.

## Important
- Be sure to disable your Gradient checking code before training your classifier. If you run numerical gradient computation on every iteration of Gradient Descent (or in the inner loop of costFunction(...)) your code will be very slow.

# Random Initialization
## Initial Value of θ
For Gradient Descent and Advanced Optimization method, need initial value for θ.

```
optTheta = fminunc(@costFunction, initialTheta, options)
```

Consider Gradient Descent:
We need to initialize theta to something and then we can slowly take steps to go downhill using gradient descent to minimize the function J(θ). So what can we set the initial vaue of theta to? Is it possible to set initial value of theta to vector of all zeros.

set initialTheta = zeros(n, 1)?

Whereas this worked okay when we were using Logistic Regression, initializing all of your parameters to zero actually doesn't work when you are training a Neural Network. Consider training the following Neural Network and let's say we initialize all the parameters of the Network to zero. And if you do that, then what that means is that at the initialization, the blue weight (colored in blue) is gonna be equal to that weight, so they're both zero. The weight colored in red and green also gonna be equal and equal to zero.

So both hidden units a<sub>1</sub><sup(2)</sup>, a<sub>2</sub><sup(2)</sup>, are going to be computing the same function of our inputs and thus you endup with for every one of your training examples, you endup with a<sub>1</sub><sup(2)</sup> = a<sub>2</sub><sup(2)</sup>.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1ec0bce7-53f1-43ff-8174-c7918e9e0c22)

And because these outgoing weights (colored in cyan) are same, you can also show that the delta values are also gonna be the same:
δ<sub>1</sub><sup(2)</sup> = δ<sub>2</sub><sup(2)</sup>

And if you work through the math further, you can show that the partial derivative with respect to your parameters will satisfy the following:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/28c2db65-0be2-4154-91ce-0fe49446d1cb)

So what this means is that even after say one Gradient Descent update, you are going to update say the first blue weight with learning rate times the derivative in LHS above and similarly second blue weight with learning rate times the derivative in RHS.

Which means that even after one Gradient Descent update those two blue weights, those two blue color parameters will end up the same as each other.

θ<sub>01</sub><sup(1)</sup> = θ<sub>02</sub><sup(1)</sup>

These will be non-zero values but will be equal to each other and same is the case with red and green weights. So you find that even after one iteration of Gradient Descent your both hidden units are computing the same functions of the inputs. So you still have:
a<sub>1</sub><sup(2)</sup> = a<sub>2</sub><sup(2)</sup>

Which means that your Neural Network can't compute really interesting functions. Imagine that you not only had two hidden units but many hidden units then what it's saying is all your hidden units are computing the exact same feature. All of your hidden units are computing the exact same function of the input and this is highly redundant representation and it prevents your Neural Network from doing something really interesting.

In order to get around this problem, the way we initialize the parameters a Neural Network therefore is with random initialization.

# Random Initialization: Symmetry Breaking
The problem we saw above is called Problem of Symmetric Weights. So the Random Initialization is how we perform symmetry breaking. So what we do is, we initialize each value of θ i.e. θ<sub>ij</sub><sup>(l)</sup> to a random value in [ -ε, ε ] (i.e. -ε <= θ<sub>ij</sub><sup>(l)</sup> <= ε)

E.g. code in Octave:
```
Theta1 = rand(10, 11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(1, 11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

rand(10, 11) gives random 10 x 11 matrix and all the values are between 0 and 1. This epsilon is unreleated to the one we were using while Gradient Checking.

# Putting It Together
## Training a Neural Network
When training a Neural Network, the first thing you need to do is pick some network architecture. And the architecture we mean connectivity pattern between Neurons.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/45186abc-ff22-4a6c-838e-44fad3a37610)

So, how do we make this choices of how many hidden layers, how many hidden units etc.?
- The number of input units: Dimension of features x<sup>(i)</sup>
- The number of output units: Number of classes
- Reasonable default: 1 hidden layer, or if > 1 hidden layer, have same no. of hidden units in every layer (usually the more the better, however the more hidden layer it's going to be more computationally expensive). Usually the number of hidden units in each layer will be may be comparable to the dimension of x i.e. comparable to the number of features, or it could be anywhere from same number of hidden units as input features to may be twice or three or four times of that.

## Training a Neural Network
- Randomly initialize weights.
- Implement Forward Propagation to get h<sub>θ</sub>(x<sup>(i)</sup>) for any x<sup>(i)</sup>
- Implement code to compute cost function J(θ)
- Implement Backward Propagation to compute partial derivatives ![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d77c5493-5d3e-48ee-89f8-68805d041bfa)
  - for i = 1:m
        Perform forward propagation and back propagation using example (x<sup>(i)</sup>, y<sup>(i)</sup>)
        (Get activations a<sup>(l)</sup> and delta terms δ<sup>(l)</sup> for l = 2,..., L)
  - There are more complicated ways to do this without for loop.
- Use Gradient Checking to compare ![image](https://github.com/vivekprm/coursera-ml/assets/2403660/66c8c9b6-6e58-4316-b567-a1eb81adae36) computed using back propagation vs using numerical estimate of gradient of J(θ). Then disable Gradient checking code.
- Use Gradient Descent or Advanced Optimization method with backpropagation to try to minimize J(θ) as a function of parameters θ.
  - For Neural Networks this function J(θ) is non-convex and so it can theoretically be susceptible to local minima and infact algorithms like Gradient descent and the Advanced optimization methods can, in theory get stuck in local optima. But it turns out in practice it's not usually a huge problem, eventhough we can't guarantee that these algorithms will find a global optimum, usually algorithms like Gradient Descent will do a very good job minimizing this cost function J(θ) and get to a very good local minimum even if it doesn't get to the global Optimum.
 
### Intuition: Gradient Descent for Neural Network
We are pretending we have only two parameters as it's easier to draw.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8f7e6b3e-fc56-4af3-b1ae-483e6a746426)

# Application of Neural Network
## Autonomous Driving
Getting a car to learn to drive itself.
