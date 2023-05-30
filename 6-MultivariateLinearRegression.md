E.g. Instead of size of the house we can have many other features that can be used to predict the price of a house.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ec76ac1c-eb02-4e01-8ba2-d77645c9e3bb)

Notation:
n = number of features, in above case n = 4
x<sup>(i)</sup> = input (features) of i<sup>th</sup> training example. E.g in this case X<sup>(2)</sup> is:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/b1ff073b-27b2-4084-bd16-0366e13f37fd)

x<sub>j</sub><sup>(i)</sup> = value of feature j in i<sup>th</sup> training example. E.g. X<sub>3</sub><sup>(2)</sup> = 2

# Hypothesis
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/15b8666b-4b3e-4468-b336-1f53f2a19d13)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e828985c-fc85-497a-99de-0176be196f89)

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume x<sub>0</sub><sup>(i)</sup> = 1 for (i belongs to 1, ...., m).
This allows us to do matrix operations with theta and x. Hence making the two vectors 'θ' and x<sup>(i)</sup> match each other element-wise (that is, have the same number of elements: n+1).]

# Gradient Descent for Multiple Variables
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2b83fde3-9b0c-4a2e-a559-1a2cbab94795)

Where X<sub>0</sub> = 1

Gredient descent looks like this:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/19ca596d-5549-472b-a9bf-8b2c215d86ba)

let's see how this partial derivative looks like:
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/8d285975-0d1e-43e3-b916-2d10837d0407)

Some of the practical tricks to make Gradient Descent work well:
# Gradient Descent in Practice I - Feature Scaling
If you have a problem where you have multiple features, if you make sure the features are on similar scale, then Gradient Descent may converge more quickly.
E.g. x<sub>1</sub> = size (0-2000 feet<sup>2</sup>)
     x<sub>2</sub> = number of bedrooms (1 - 5)
     
If we plot cost function for such function, it will be very long skinny plot.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d921ee48-52ad-4b9f-938d-1150108f065d)

It's going to take much larger time in converging.

In this setting useful thing to do is to scale the features.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f5e1ba24-0942-4069-b3e2-2b187dcf7aeb)

For feature scaling:
- Get every feature into approximately a -1 <= x<sub>i</sub> <= 1 range

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/124ff497-83e7-4c3b-9541-72ea3794a603)

## Mean Normalization
Replace x<sub>i</sub> with x<sub>i</sub> - μ<sub>i</sub> to make features have approximately zero mean (Do not apply to x<sub>0</sub> = 1)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5b61a9de-91bd-4ced-ba69-4f4d8bdf5fbc)

# Gradient Descent in Practice II - Learning Rate
