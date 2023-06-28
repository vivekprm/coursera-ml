# Deciding What to try Next
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

Fortunately there is a pretty simple technique 
