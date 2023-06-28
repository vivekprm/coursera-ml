![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c93ea8d6-c742-43ee-a49b-0ed56d80c573)# Deciding What to try Next
Suppose you are developing a Machine Learning System or trying to improve the performance of a Machine Learning System, how do you go about deciding what are the promising avenues to try next. To explain this let's continue our example of learning to predict housing prices and let's say you implemented Regularized Linear Regression.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ff0e17fa-5c66-4b01-96d1-a977af0e0b74)

Now suppose that after you take your learned parameters, if you test your hypothesis on the new set of houses, you find that it makes unacceptably large errors in its predictions. What should you try next in order to improve the learning algorithm?

Few things you could try:
- Get more training examples.
  - Sad thing is, sometimes getting more training data doesn't help.
- Try smaller set of features to prevent overfitting.
- Get additional features. May be current set of features are not informative enough.
- Try adding polynomial features (x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, x<sub>1</sub>x<sub>2</sub>, etc.)
- Try decreasing λ
- Try increasing λ

Each of these measures in itself can be huge projects and may need significant amount of time. So it would be helpful to know in advance if it will help.
Unfortunately the most common method that people use is, pick one of these is to go by gut feeling. And spend lot of time collecting the detail only to find out later that it may not work.

Fortunately there is a pretty simple technique that can let you quickly rule out half of the things on this list as being potentially promising thing to persue. And there is a very simple technique, that if you run, can easily rule out many of these options and potentially save you a lot of time persuing something that's just not going to work.

# Machine Learning Diagnostic
A test that you can run to gain insight what is/isn't working with a learning algorithm, and gain guidance as to how best to improve it's performance. Diagnostics can take time to implement, but doing so can be a very good use of your time.

# Evaluating a Hypothesis
When we fit the parameters of our learning algorithm, we think about choosing the parameters to minimize the training error. One might think that getting a really low value of training error might be a good thing but we hae already seen that just because a hypothesis has low training error that doesn't mean it is necessarily a good hypothesis and we have already seen the example of how a hypothesis can overfit.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c30fd524-8a4a-44bc-87d6-aba57d04c7c1)
he 0/1 misclassification eror.
Therefore it fails to generalize to new examples not in training set. In this simple example we could plot the hypotesis and could see what's going on. But in general problems with more features than just one feature (like below)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/0bf6264a-ab09-42ab-9ef8-6e2e1f22c85f)

For problems with large number of features like these it becomes hard or may be impossible to plot what the hypothesis looks like and so we need some other way to evaluate our hypothesis.

The standard way to evaluate a learned hypothesis is as follows. Suppose we have a dataset like this:

Here we have shown just 10 training examples but we many have hundreds or may be thousands of training examples. In order to make sure we can evaluate our hypothesis, what we are going to do is split the data we have into two portions. The first portion is going to be our usual training set and the second portion is going to be our test set. A pretty typical split of this all the data we have into a training set and test set might be around 70%, 30% split.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1c4e8629-03f9-4ad8-b577-f45c85d26cc2)

If theser is any sort of ordering to the data it's actually better to better to send a random 70% of your data to training set and a random 30% to test set.

## Training/testing procedure for Linear Regression
- Learn parameter θ from training data (minimizing training error J(θ))
- Compute test set error denoted as J<sub>test</sub>(θ)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/9566e254-7d56-4974-8334-fbe5290dfc8b)

This is the definition of test set error if you are using the Linear Regression. If we are having classification problem and using Logistic Regression test set error can be represented as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1b8f12dd-19be-4fa2-bc93-49b96112e9f0)

While definition of test set error J<sub>test</sub> is perfectly reasonable sometimes there is an alternative test sets metric that might be easier to interpret and that's the misclassification error. It's also called the 0/1 misclassification error. Where 0/1 denoting either you get an example wrong or right.

Below is error in a prediction as:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3d639d86-acd3-45ee-8f43-b2ebef87d81d)

We could then define the Test error as:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/396cf414-529b-45f6-8c7d-08f26b81cfcc)

# Model Selection and Train/Validation/Test Sets
Suppose you are left to decide what degree of polynomial to fit to a dataset, what features to include that gives you a learning algorithm. Or suppose you would like to choose the regularization parameter λ for learning algorithm. How to do that? These are called model selection problems.

We've already seen a lot of times the problem of overfitting. More generally that's why training set error is not a good predictor for how well the hypothesis will do on new examples.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/173f9e99-0d42-43bf-99c7-3aca17aeb962)

Now let's consider the model selection problem. Let's say you are trying to choose what degree polynomial to fit to data:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/fa50f36d-9c05-469d-b403-903117c91759)

So it's as if there is one extra parameter in this algorithm denoted by d which is degree of polynomial. So along with theta there is one more parameter d that we are going to evaluate using our dataset.

What you could do is: take your first model and minimize the training error and this will give you some parameter vector theta (θ<sup>(1)</sup>) and then you could take your second model and get some other parameter vector theta (θ<sup>(2)</sup>) and so on.

Then we take these parameters and look at the test errors J<sub>test</sub>(θ<sup>(1)</sup>), J<sub>test</sub>(θ<sup>(2)</sup>) and so on. Now omne thing we could do then  is, in order to select one of these models, we could then see which model has lowest test set error.

Let's just say for this example we ended up choosing the fifth order polynomial. So this seems reasonable so far. But now let's say I want to take my fifth hypothesis and let's say I want to ask How well does this model generalize?

One thing we could do is look at how well our fifth order polynomial hypothesis had done on my test set i.e. J<sub>test</sub>(θ<sup>(5)</sup>)

But the problem is this will not be a fair estimate of how well my hypothesis generalizes and the reason is what we've done is we've fit this extra parameter d that is degree of polynomial and we have fit that parameter d using the test set, namely we chose the value of d that gave us the best possible performance on the test set and so the performance of my parameter vector θ<sup>(5)</sup>, on the test set is likely to be an overly optimistic estimate of generalization error.

To address this problem in model selection setting if we want to evaluate a hypothesis, this is what we usually do instead. Given a dataset instead of just splitting into training, test set  what we are going to do is split it into three parts First part is going to be called training set, second part is called cross validation set 
and the last part is called usual test set. 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/a35f9e60-b82e-42e5-8af5-e3b5e5fb48b3)

Pretty typical ratio at which to split these things will be 60%, 20%, 20%. These numbers can vary a little bit.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/060b220b-b8d2-46bc-8d10-d06786a92700)

So now that we've defined training, validation and test set, We can also define the training error, cross validation error, and test error as below:We can also define the training error, cross validation error, and test error as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/923a65ca-df98-480c-9f4e-f8a12321b2e5)

So when facedwith a model selection problem like this, what we are going to do is, instead of using the test set to select the model, we're instead going to use the cross validation set to select the model. And we are going to pick the hypothesis with lowest cross validation error.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/53aff99b-2012-43be-bd29-d67866a25b74)

Let's say in our case 4th model was with lowest cross validation error.
