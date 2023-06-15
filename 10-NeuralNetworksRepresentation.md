# Non-linear Hypotheses
Consider a Supervised Learning Classification problem where we have a training set as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/7583dfa6-2902-485f-a1f3-5281149c5110)

If you want to apply Logistic Regression to this problem, one thing you could do is apply Logistic Regression with a lot of non-linear features like that:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a9d19ffd-b287-491e-b268-21460bf84b80)

And then you may have a plot which fit the training set like this:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/498064b2-2efd-4a52-a208-736b78ca7c2d)

This particular method works well when let's say you only have two features - x1 and x2, because then you can include all those polynomial terms of x1 and x2.
But for many interesting machine learning problems would have a lot more features than just two.

So e.g. you have been given a set of features for houses and you have to predict whether it will be sold in next few month. So it's a classification problem. And we 
can have 100s of features like this:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f840a173-e437-454b-b6fa-0301f23645f9)

Just second order terms will be 5000. So no of quadratic terms incresed by O(n<sup>2</sup>). So, including all the quadratic features doesn't seem like a good idea 
becuase that's a lot of features and you might endup over fitting the training set and it can also be computationally expansive to be working with these many features.

One thing you could do is, include a subset of these, so if you could include only the features x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, 
x<sub>3</sub><sup>2</sup>,...., x<sub>100</sub><sup>2</sup>. Then the number of features is much smaller. Here you have ony 100 such quadratic features, but this is
not enough features and certainly won't let you fit the dataset like that on the upper left. Infact if you include only these quadratic features together with the
original x1, x2, ..., x100 features then you can't actually fit very interesting hypotheses. So you can fit things like you know, access a line of ellipses like 
these, you certainly can't fit a more complex dataset like in this case.

So 5000 features seeems like a lot, if you were to include the cubic or third order polynomial features e.g. x<sub>1</sub>x<sub>2</sub>x<sub>3</sub>, 
x<sub>1</sub><sup>2</sup>x<sub>2</sub>, x<sub>10</sub>x<sub>11</sub>x<sub>17</sub>, .... etc. You can imagine there are gonna be a lot of these features. It's going to
be of O(n<sup>3</sup>) order. Which blows up your feature space dramatically and this doesn't seem like a good way to comeup with additional features with which to 
build none many classifiers when n is large.

For many machine learning problems, n will be pretty large. Here is an example:

Let's consider the problem of computer vision and suppose you want to use machine learning to train a classifier to examine an image and tell us whether or not the 
image is a car. Many people wonder why computer vision could be difficult. When you and I look at this picture it's so obvious what this is. You wonder how is it that
a learning algorithm could possibly fail to know what this picture is.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6f55fa22-3ea4-4f0c-a162-452127cae534)

To know why computer vision is hard let's zoom into a small part of the image like the area where little red rectangle is. It turns out where you an I see a car, 
a computer sees that (See below)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/4a558c87-90eb-4110-ac10-3f8a17997467)

What it sees is this matrix or this grid of pixel intensity values that tells us the brightness of each pixel in the image. So the computer vision problem is to look
at this matrix of pixel intensity values, and tell us that these numbers represent the door handle of a car.

Concretely , when we use machine learning to build a car detector, what we do is we comeup with a label training set with let's say a few label examples of cars and 
a few label examples of things that are not cars, then we give our training set to the learning algorithm train a classifier and then, you know, we may test it and
show the new image and ask "What is this new thing?"

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/eaad130c-5129-408e-8eaf-457ebbdce52c)

To understand why we need non-linear hypothesis let's take a look at some of the images of cars and may be non cars that we might feed to our learning algorithm.
Let's pick a couple of pixel locations in our images, so that's pixel 1 location & pixel 2 location 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/87065820-7b93-48e5-a8b0-93770f0212de)

And let's plot this car, you know, at the location, at a certain point depending on the intensities of the pixel 1 and pixel 2. Let's do that with few other images
of cars and non-cars:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9d0fbf83-3a70-41e8-a801-6d138f01f99a)

What we will find that the cars and non-cars endup lying in different regions of the space and what we need therefore is some sort of non-linear hypothesis to try to 
separate out the two classes.

What is the dimension of the feature space?
Suppose we were to use just 50 by 50 pixel images Then we would have 2500 pixels, and so the dimension of our feature size will be n = 2500, where our feature vectox x
is a list of all the pixel intensities. 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6a372753-8abe-43a8-bb37-17f7f382e354)

Each of these have value from 0-255. So we have n=2500 when we are using gray scale images. If we were using RGB then we will have n=7500. 

So, if we were to try to learn a non-linear hypothesis by including all the quadratic features i.e. all the terms of the form X<sub>i</sub> x X<sub>j</sub>.
Well with the 2500 pixels we would end up with a total of three million features and that's just too large to be reasonable; the computation would be very expensive 
to find and to represent all of these three million features per training example.

So, Simple Logistic Regression together with adding in may be the quadratic or the cubic features that's just not a good way to learn complex non-linear hypothesis 
when n is large because you just endup with too many features. 

Neural Networks are much better way to learn Complex Hypothesis, Complex non-linear hypothesis even when your input feature space, even when n is large.

# Neurons & the Brain
Neural Networks are pretty old algorithm that was originally motivated by the goal of having machines that can mimic the brain.

## Neural Networks
- Origins: Algorithms that try to mimic the brain. Was very widely used in 80s and early 90s; popularity diminished in late 90s.
- Recent Resurgence: State-of-the-art technique for many applications.
  - One of the reason of resurgence is: Neural Network algorithms are pretty complex algorithms, very recently computers became fast enough to really run large scale Neural Networks and because of that as well as few other technical reasons modern Neural Networks today are the State of the Art technique for many applications.  

So when you think about mimicking the brain, while human brain does so many amazing things. It can learn to see process images, learn to hear, learn to process our senses of touch, learn to do math, learn to do Calculus and the brain does so many different and amazing things. It seems like if you want to mimic the brain it seems
like you have to write lots of different pieces of Software to mimic all of these different fascinating, amazing things that the brain does. 

But does this fascinating hypothesis that the way brain does all of these different things is not worth like a thousand different programs, but instead the way the brain does it is worth just a single learning algorithm.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f39756f4-be39-4cde-8db5-3b392d7a3e41)

This is just a hypothesis but let's see some evidence of this. Red part of the brain in the picture above is your Auditory Cortex, and the way you are understanding my voice is, your ear is taking the sound signal and routing the sound signal to your Auditory Cortex and that's what allowing you to understand my words.

Neuroscientists have done the following fascinating experiments where you cut the wire from the ear to the auditory cortex and you re-wire in this case an animal's brain so that the signal from the eyes, from the optic nerve eventually gets routed to the Auditory Cortex. If you do this it turns out, the Auditory Cortex will learn to see.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/df678e04-9fef-400b-adb6-6085f7285826)

And this is in every single sense of the word see as we know it. So, if we do this to animals, the animals can perform visual discrimination task and they can look at images and make appropriate decision based on the images and they are doing it with that piece of brain tissue.

Here is another example, 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/11a6b803-3d55-4251-ac70-8cc0c3975df3)

That red piece of brain tissue is your Somatosensory Cortex. That's how you process your sense of touch. If you do a similar rewiring process then the Somatosensory cortex will learn to see.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/fb6c6f66-a10c-43ea-86a1-8cc62f702f0f)

Because of this and other similar experiments (these are called Neuro Rewiring experiments), there is this sense that if the same piece of physical brain tissue can process sight or sound or touch then may be there is one learning algorithm that can process sight, sound or touch. And instead of needing to implemnet thousand different programs or a thousand different algorithms to do, you know, the thousand wonderful things that brain does, maybe what we need to do is figure out some approximation or to whatever the brain's learning algorithm is and implement that and let the brain learn by itself how to process these different types of data.

# Sensor Representations in the Brain

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3c8130d7-54f5-415b-afdf-b2eb22cb657b)

# Model Representation I
Neural Networks were developed as simulating neurons or networks of neurons in the brain. Below is a single nuron:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/793e1825-2c4c-4fa3-b86a-742457045660)

Our brain is jam packed full of neurons like these and neurons are cells in the brain. Neuron has a Cell body and neurons have number of input wires and these are called the "dendrites". You think of them as input wires and these receive inputs from other locations. Neurons also have output wire called "Axons". These are used to send signals to other neurons.

The way that neurons communicate with each other is with little pulses of electricity, they are also called Spikes but that just means pulse of electicity.

## Neuron Model: Logistic Unit
In a neuro network, or rather, in an artificial neuron network that we've implemented on the computer, we are going to use a very simple model of what a Neuron does. We're going to model a Neuron as just a logistic unit:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c26bde40-a5a0-4d06-bc43-14215b55bdf8)

Yellow circle in the pic above is analogous to the body of Neuron. We then feed the Neuron few inputs analogous to dendrites. Neuron does some computation and outputs some value in this case h(θ) similar to Axons.

So this diagram or model represents:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/4fc467c7-9508-49e5-b6f2-451195e0861a)

Where as usual, x and θ are our parameter vectors.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/72c76e6c-26b5-4a79-a882-d1d22449d153)

So this is very simple, may be a vastly oversimplified model of the computations that the Neuron does, where it gets a number of inputs x1, x2 & x3 and it outputs some value computed like so (in this case h(θ)).

In this case we have only drawn x1, x2 and x3 inputs. Sometimes we draw additional input x0 also called "Biased Unit" or the "Bias Neuron" but because x0 is already equal to one sometimes we draw sometimes we don't depending on whatever is more notationally convenient. 

Sometimes we will say that this is an Artificial Neuron with a Sigmoid or Logistic Activation Function. We keep continuing saying θ as prameters of the model but sometimes people also refer it as Weight of the Model.

This little diagram (see above) represents a single Neuron. What a Neural Network is, is just a group of these different Neurons strung together.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a25f83e4-fc48-45ea-b3af-2942515ba740)

Here we have input units x1, x2, x3 and here we have three Neurons which we have written a<sub>1</sub><sup(2)</sup>, a<sub>2</sub><sup(2)</sup>, a<sub>3</sub><sup(2)</sup>. We can also add a Bias Unit a<sub>0</sub><sup(2)</sup> which always outputs one. And finally we have third node at the final layer and this is the third node that outputs the value that the hypothesis h(x) computes.

First layer in Neural Network is also called "Input Layer". The final layer (layer 3 in this case) is also called the "Output Layer". Layer 2 in between is also called "Hidden Layer". Neural Networks can have more than one Hidden Layer.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/026de983-9412-4c42-8ba3-d3cad233d01c)

To explain specific computations represented by a Neural Network. Here is bit more notation:

a<sub>1</sub><sup>(j)</sup> = "activation" of unit i in layer j
θ<sup>(j)</sup> = Matrix of weights controlling function mapping from layer j to layer j+1

So here are the computations represented by the above diagram:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6e248941-0cf3-4ca8-8c77-36c99be62835)

So in this case θ<sup>(1)</sup> will be 3x4 dimension matrix. And more generally:

If a network has s<sub>j</sub> units in layer j and s<sub>j+1</sub> units in layer j+1, then θ<sup>(j)</sup> will be of dimension s<sub>j+1</sub>xs<sub>j</sub>+1
E.g. in pic below θ<sup>(1)</sup> is 4 x 3 dimensional matrix as s<sub>j+1</sub> = 4 and s<sub>j</sub> = 2

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d7ec5479-aae4-49cf-8ab9-f62841c3d3f6)

# Model Representation II
## Forward Propagation: Vectorized Implementation
Consider this Neural Network:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/dc33f6dd-398d-46c8-abb3-38312eb8e77e)

Previously we said that sequence of steps that we need in order to compute the output of the hypothesis are the equation above. Now we are going to define some extra terms. We're going to define a new variable z<sub>k</sub><sup>(j)</sup> that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c2c4d5c5-77fd-4239-94c2-10abffba422c)

In our case x and z<sup>(2)</sup> will be:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ae9dfb00-5419-4e30-aadb-a699904112fa)

And resulting equation will be:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/fe86f029-500f-4858-9abf-ec79ef9b4c9d)

We can think of input layer as activation of Layer 1 a<sup>1</sup>. So now 

z<sup>(2)</sup> = θ<sup>(1)</sup> a<sup>(1)<sup>
  
Add extra bias unit as a<sub>0</sub><sup>(2)<sup> = 1. So now a<sup>(2)<sup> is 4 dimensional vector. 

And finally comput actual value of the hypothesis:
z<sup>(3)</sup> = θ<sup>(2)</sup> a<sup>(2)<sup>

h<sup>θ</sup>(x) = a<sup>(3)<sup> = g(z<sup>(3)<sup>)
  
This process of computing h<sup>θ</sup>(x) is also called forward propagation.

More generally:

In other words, for layer j=2 and node k, the variable z will be:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2556c9d0-9515-43b6-92cc-52b587bcae05)

The vector representation of x and z<sup>j</sup> is:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8a595c4a-c500-4f7b-a798-75a76123f8a8)


## Neural Network Learning it's own features
This forward propagation view also helps us to understand what Neural Networks might be doing and why they might help us to learn interesting non-linear hypothesis.
Consider following Neural Network:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/54db2868-f27d-402b-b408-8379cffa2daf)

Let's say we coverup left part of the picture for now. If you look at what's left in this picture. This looks a lot like Logistic Regression  :

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/eba2b840-e017-4318-8083-5e1a6964c9fa)

What this Neural Network is doing, is just like logistic regression, except that rather than using the original features x1, x2, x3 is using these new features a<sub>1</sub><sup>(2)</sup>, a<sub>2</sub><sup>(2)</sup>, a<sub>3</sub><sup>(2)</sup>.
  
The cool thing about this is that features a<sub>1</sub><sup>(2)</sup>, a<sub>2</sub><sup>(2)</sup>, a<sub>3</sub><sup>(2)</sup>, they themselves are learned as functions of the input. Concretely the function mapping from Layer1 to Layer2 that is determined by some other set of parameters Θ<sup(1)</sup>. So as if the Neural Network, instead of being constrained to feed the features x1, x2, x3 to Logistic Regression. It gets to learn it's own features a<sub>1</sub><sup>(2)</sup>, a<sub>2</sub><sup>(2)</sup>, a<sub>3</sub><sup>(2)</sup> to feed into the Logistic Regression and as you can imagine depending upon what parameters it chooses for Θ<sup(1)</sup>, you can learn some pretty interesting and complex features and therefore you can end up with a better hypothesis than if you were constrained to use the raw features x1, x2, x3 or if you were constrained to say choose the polynomial terms x1x2, x2x3, .. and so on. 

But instead this algorithm has the flexibility to try to learn whatever features at once using these a1, a2, a3 in order to feed in to this last unit that's essentially a Logistic Regression unit.

# Other Network Architecture
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ec62ab73-9a7e-4462-aa68-ee2c0b06a36e)

You can have different types of Neural Network diagrams. And the way Neural Networks are connected is called Architecture.
  
Consider below Neural Network:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/db101d4c-792e-4fa2-a493-535a4022f41e)

Let a<sup>(1)</sup> = x belongs to R<sup>n+1</sup> denote the input (with a<sub>0</sub><sup>(1)</sup> = 1). How would you compute a<sup>(2)</sup>?
Ans: ![image](https://github.com/vivekprm/coursera-ml/assets/2403660/71fb3667-7480-4acc-aa3e-87c03d2eab31)

# Examples & Intuitions I
## Non-linear classification Example: XOR/XNOR
Consider the following example where we have features X1 and X2 that are binary values. In this case we have two positive examples and two negative examples. You can think of it as a simplified version of a more complex learning problem. 

And what we would like to do is learn a non-linear division of boundary that may need to separate the positive and negative examples.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/30fa61bc-00aa-40e6-b5eb-d779a1189ad6)

So how can a Neural Network do this and rather than using the example and variable to use this may be easier to examine example on the left. Concretely, what it is really computing the type of label:

y = x1 XOR x2
    X1 XNOR x2
    NOT (x1 XOR x2)

So we are going to have positive examples of either both are true or both are false and we have y = 0 when only one of them is true.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/7e9a275d-8ab4-4eae-af98-2299c967a23c)

We are going to figure out if we can get a Neural Network to fit to this sort of training set.

In order to build up to a network that fits the XNOR example we're going to start with a slightly simpler one and show a network that fits the AND function.

## Simple Example: And 
Concretely, let's say we have input x1 and x2 that are again binaries and let's say our target labels y = x1 AND x2 i.e. logical AND

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1ee53a42-00a2-4acb-bc78-e11bf26710dd)

So, can we get a one unit network to compute this Logical AND function?

In order to do so we are also going to draw the Bias Unit. Let's assign some values to the Weights or parameters of this Network.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b06aa8bc-f7b0-41e2-beb4-6c183251e4b3)

So now our hypothesis looks like below:

h<sub>θ</sub>(x) = g(-30 + 20 x x1 + 20 x x2)

θ<sub>10</sub><sup>(1)</sup> = -30
θ<sub>11</sub><sup>(1)</sup> = 20
θ<sub>12</sub><sup>(1)</sup> = 20

Lets look at what this little single Neuron will compute. Just to remind this is how sigmoid function looks like:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/44335c20-4882-4982-adf5-9543d093ab58)

Hypothesis result looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/cb361988-035b-476e-b33d-747bcd34d124)

And if you look at the hypothesis result it's computing the AND function.

## Example: OR Function

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0041f6c8-122f-4b1e-bf6f-29f57a421a37)


# Examples and Intuitions II
We can also have networks to compute negation.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3603ec3a-dabf-4e47-b921-4dc4f653ac64)

So idea is put large negative weight in front of x1.

Putting it together : x1 XNOR x2

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a1d1cff2-307e-445e-afb6-f0594edb04f5)

E.g. Handwritten digit classification by Yann LeCun

# Multiclass Classification
Handwritten digit classification problem is actually Multiclass Classification Problem because there were ten possible categories for recognizing the digits from 0 through 9.

The way we do Multiclass Classification in a Neural Network is essentially an extension of One-vs-All method.

## Multiple Output Units: One-vs-all
Let's say we have a computer vision example where instead of just trying to recognize cars, we are trying to recognize 4 categories of objects (Padestrian, Car, Motorcycle, Truck).

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/dbd4489c-4dbf-4657-b992-d5e94a072642)

We will build a Neural Network with 4 output units so that our Neural Network now outputs a vector of four numbers. First output unit is going to classify if the image is Padestrian. 2nd one classify Cars and so on.
