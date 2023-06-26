![image](https://github.com/vivekprm/coursera-ml/assets/2403660/99f67db9-a421-4b3b-ab5b-219112607447)# Neural Networks: Cost Function
We are going to focus on application of Neural Networks to Classification problems. So suppose we have a network and m training sets as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8f8261eb-7ca0-41ac-8722-bca5eaa5b181)

We are using L to denote the total number of layers in this network. So for the network shown on the left we would have L = 4
s<sub>l</sub> to denote the number of units (not counting bias unit) in layer l.

E.g. s<sub>1</sub> = 3, s<sub>2</sub> = 5, s<sub>4</sub> = s<sub>L</sub> = 4

We are going to consider two types of classification problem:
- **Binary Classification**: Where the labels y are either 0 or 1. In this case we will have one output unit. So this Neural Network has 4 output units, but if we had binary classification we would have only one output unit that computes h<sub>θ</sub>(x) and the output of the Neural Network h<sub>θ</sub>(x) is going to be a real number.In this case s<sub>L</sub> = 1. To simplify notation in this case we are also considering K = 1. You can think of K as also denoting the number of units in the output layer.
- **Multi-class Classification**: Here we may have K distinct classes. In this case we will have K output units and hypothesis h<sub>θ</sub>(x) is K dimensional vector. And number of output units s<sub>L</sub> = K and usually we will have K >= 3. Because if we had two classes, then we don't need to use One-vs-All method, we need to use One-vs-All method only if we have K >= 3 classes, If we had only two classes we only need to use one output unit.

Now let's define the Cost Function for our Neural Network.

## Cost Function
The Cost Function we use for the Neural Network is going to be a generalization of the one that we use for Logistic Regression.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b519fdd1-63e5-4516-b527-df88652342ff)

For Neural Network instead of having just one output unit, we may instead have K of them. So here is our Cost Function:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/55673bc9-0278-4c97-af9c-27af42d5f042)

## Backpropagation Algorithm
It's an Algorithm to minimize the Cost Function. Here is the Cost Function that we wrote:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f73c4281-701f-4f33-926c-d30baa5a87b1)

Now our job is to minimize this cost function J(θ). What we need to do therefore is to write code that takes below inputs:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9e556f0a-7e0c-435a-b023-92e84730fe27)

Remember the parameter in the Neural Network θ<sub>ij</sub><sup>(l)</sup> is real number.

### Gradient Computation
Let's talk about the case when we have only one training example. So imagine, if you will that our entire training set comprises only one training example which is a pair (x, y). Let's tap through the sequence of calculations we would do with this one training example. First thing we do is we apply forward propagation in order to compute whether the hypothesis actually outputs given this input x.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3ca6171d-2217-4ddd-b8b6-e7cebd67e42e)

Next, inorder to compute the derivatives, we are going to use an Algorithm called Back Propagation.

### Gradient Computation: Back Propagation Algorithm
Intuition: δ<sub>j</sub><sup>(l)</sup> = "error" of node j in layer l

The intuition of the back propagation algorithm is that for each node we are going to compiute the term δ<sub>j</sub><sup>(l)</sup>, that's going to somehow represent the error of note j in the layer l. Recall that a<sub>j</sub><sup>(l)</sup> is the activation of jth unit in layer l and so this δ term is in some sense going to capture our error in the activation of Node. Concretely, taking the example of Neural Network that we have (below) which has four layers and so L=4.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/292b7438-733d-4a09-82c7-08a683dae195)

For each output unit we are going to compute this δ term. So e.g. δ of 4th layer.

δ<sub>j</sub><sup>(4)</sup> = a<sub>j</sub><sup>(4)</sup> - y<sub>j</sub>

where y<sub>j</sub> is actual value in our training example.

a<sub>j</sub><sup>(4)</sup> can also be written as (h<sub>θ</sub>(x))<sub>j</sub>

By they was if you think of δ, a & y as vectors, then you can also take this and comeup with a vectorized implementation of it. Which is:

δ<sup>(4)</sup> = a<sup>(4)</sup> - y

Dimension of δ, a & y vecotrs, is number of output units in our network.

Next we compute the δ term for the earlier layers in our network.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9bcde8e5-708c-46e5-8964-71dbdfea764a)

Where .* signifies elementwise multiplication. 
Term g'(z<sup>(3)</sup>) is actually the derivative of the activation function g evaluated at the input values given by z<sup>(3)</sup>.

When you compute g'(z<sup>(3)</sup>) derivative term it just computes to:

g'(z<sup>(3)</sup>) = a<sup>(3)</sup> .* (1 - a<sup>(3)</sup>)

Where a<sup>(3)</sup> is vector of activations 1 is vector of ones.

There is no δ<sup>(1)</sup> term because first layer corresponds to the input layer and that's the feature that we observed in our training set, so that doesn't have any error associated with that.

The name Back Propagation comes from the fact that we start by computing the δ term for the output layer and then we go back a layer and compute the δ term for previous layer and so on.

Finally, the derivation is surprisingly complicated, surprisingly involved. But if you do these few steps of computation it's possible to prove if you ignore regularization.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/17ddff12-774d-49a9-9976-b81e418a25b3)

Let's take everything and put all of this together to talk about how to implement back propagation to compute derivatives with respect to your parameters and for the case when we have a large training set not just a training set of one example.

### Back Propagation Algorithm
Suppose we have a training set of m examples like shown below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/87e041a6-666b-4d05-baa3-45faf8d774c2)

The first thing we are going to do is:

Δ<sub>ij</sub><sup>(l)</sup> = 0 (for all l, i, j)

These Δ are going to be used as accumulators and will slowly add things to compute these partial detivative ![image](https://github.com/vivekprm/coursera-ml/assets/2403660/7f1c716a-cd92-4619-ba70-2f1f24eca469)

Next we are going to loop through our training set :

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/91b9e84f-e198-420f-92a4-e72aecd7f628)

Lat equatation can be vectorized as:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3d157d66-5469-478c-909c-77998db73062)

Finally after executing the body of the for loop we then go outside the for loop and compute the following:

![image](https://github.com/vivekprm/coursera-ml/assets/240 3660/1ae2f609-cbb8-4f03-84cb-64bc3c28d596)

So complete algorithm is as follows:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/79ca1c7b-5543-471e-a2bf-42e89359c863)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/bd44de52-bd07-4b33-abc2-ff9007c0c964)

# Backpropagation Intuition
Let's look at little bit more at the mechanical steps of back propagation, and try to give you a little more intuition of what mechanical steps of Back Propagation is doing to convenience you that, it's atleast a reasonable algorithm.

In order to understand Back Propagation let's take another closer look at what forward propagation is doing. Here is a Neural Network with two input units and two hidden units and then finally one output unit.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/408b60a1-b6db-4cdb-a4c3-39d9eef69457)

Back propagation is doing a process very similar to this except that instead of the computations flowing from the left to the right of this network, flows right to left of the network.

To understand better what Back Propagation is doing. Let's look at the Cost Function that we had with only one output unit:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c6aa263d-1593-483a-a4fb-d2adc4c06b49)

Focusing on a single example x<sup>(i)</sup>, y<sup>(i)</sup>, the case of one output unit, and ignoring regularization (λ = 0)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b2f0f7c0-3682-4bb8-bbac-d03303057621)

i.e. how well is the network doing on example i?

Now let's look at what Back Propagation is doing. One useful intuition is that BackPropagation is computing these δ<sub>j</sub><sup>(l)</sup>, we can think of these as the error of the activation value that we got for unit j in the l<sup>th</sup> layer.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6a9c4fa8-f123-402b-b999-bc7562a998f2)

So concretely Cost Function is a function of label y and h<sub>θ</sub>(x) output value. And if we could go inside the Neural Network and change those z<sub>j</sub><sup>(l)</sup> values a little bit then that would affect these values that the Neural Network is outputting and so that will endup chaning the Cost Function.

If you know calculus, what these delta terms are is, they turn out to be the partial derivative of the Cost Function, with respect to these intermediate terms that we are computing.

And so they are measure of how much would we like to change the Neural Network's weight, in order to affect these intermediate values (z<sub>1</sub><sup>(2)</sup>, z<sub>2</sub><sup>(2)</sup>, z<sub>1</sub><sup>(3)</sup>, z<sub>2</sub><sup>(3)</sup>) of the computation, so as to affect the final output of the Neural Network  h<sub>θ</sub>(x) and therefore affect the overall cost.

But lets look in more detail about what BackPropagation is doing:
For the output layer it sets the delta term as:

δ<sub>1</sub><sup>(4)</sup> = y<sup>(i)</sup> - a<sub>1</sub><sup>(4)</sup>

It's difference between actual value of y minus what was the value predicted, and so we are going to compute δ<sub>1</sub><sup>(4)</sup> like above.

Next what we are going to do is propagate these values backward ans endup computing δ<sub>1</sub><sup>(3)</sup>, δ<sub>2</sub><sup>(3)</sup> for previous layer and finally compute δ<sub>1</sub><sup>(2)</sup>, δ<sub>2</sub><sup>(2)</sup>. 

Now let's look at how δ<sub>2</sub><sup>(2)</sup> is calculated, so will just label the weights as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/53257ef3-cc7a-44b6-9485-6f7b6be013a7)

So this δ<sub>2</sub><sup>(2)</sup> is computed as wieghted sum of these delta values by the corresponding edge strength.

Similarly δ<sub>2</sub><sup>(3)</sup> is computed as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0230f4b1-de80-4613-a479-0df19df5c966)

