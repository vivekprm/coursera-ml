# Example 1: Suppose we want to predict housing prices

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6d5d33b6-0219-454a-afc2-800cceae23a6)

Suppose user wants to sell a house of area, 750 square feet, What should be the best prices he should get. How machine learning algorithm can help us? If we put a 
straight line through the data set we get approximate price of 150k $. But may be that’s not the only learning algorithm we can use.

In this case instead of fitting the dataset on line. We can fit it in a quadratic function, in that case may be we can sell it for 200k $. So in supervised learning 
we gave the data set with right answer to our algorithm and task of the algorithm was to predict the prices for new set of data.

**This is also called a Regression problem** i.e. we are trying to predict continuous value output.

# Example 2: Suppose by looking at the medical record we want to predict whether a breast cancer is malignant or benign.
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/15feee06-c31f-489f-b95f-b60bcc6ef367)

Suppose in somebody’s record tutor size is as represented above with pink arrow. We want to predict whether it’s malignant or benign? It’s basically a classification 
problem where we are trying to predict 0 or 1.

Another way of representing this data set is showing it on a single line with different symbol.
In other machine learning model me may have more than one feature. E.g. lets say with tumor size we have age of the patient having cancer:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/6eef6c46-388a-4ed7-9a7b-fdf131ccdbaf)

In the picture above red cross shows benign tumor and blue circles shows malignant. Lets say a patient have tumor size and age as shown with pink dot in below 
picture, predict whether it's benign or malignant.

Here algorithm tries to draw a boundary to separate malignant with benign in best possible manner...
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/98cfd236-93a0-44a1-9bae-9b5bcf1d772b)

So in this case our data belong to benign side. In other problems we can have more features. E.g. we can modify this problem and use below features:
- Clump thickness
- Uniformity of cell size.
- Uniformity of cell shape.
- ...

So how do we deal with infinite number of features. **Support Vector Machine algorithm provides a trick to deal with that**.

So, just to recap,
# Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the 
input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a 
continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict 
results in a discrete output. In other words, we are trying to map input variables into discrete categories.

## Example 1
Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression 
problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here 
we are classifying the houses based on price into two discrete categories.

## Example 2
- (a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture.
- (b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.
