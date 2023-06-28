![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ac1a2f08-9077-4faf-8f14-028d82df50d6)If you run a learning algorithm and it doesn’t do as well as you are hoping, almost all the time it will be because you have either a higher bias problem or high variance problem. In other words, either an underfitting problem or an overfitting problem. In this case, it’s very important to figure out which of these two problems is bias or variance or a bit of both that you actually have. Because knowing which of these two things is happening would give a very strong indicator for what are the useful and promising ways to try to improve your algorithm.

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
