# Basic Operations

```
a = 3
% semicolon supresses the print output.
b = 4;

c = pi;
disp(c);

disp(sprintf('2 decimals: %0.2f', c));

format long;
disp(c);

format short;
disp(c);

% Vectors & Matrices
A = [1,2; 3,4; 5,6]

R = [1, 2, 3]
V = [4; 5; 6]

% M is a row vector, starts with 1 increments 0.1 and goes to 2
M = 1:0.1:2

% N is a row vector, goes from 1 to 6
N = 1:6

% Generates 2x3 matrix of ones
ones(2,3)

W = zeros(1, 3)

% Random number using normal distribution
X = rand(1, 3)

% Random number using Guassian Distribution
Y = randn(1, 3)

% Complex expression and plotting
Z = -6 + sqrt(10) * (randn(1, 10000));
hist(Z)

% Histogram with more bins
hist(Z, 50)

% Generate 4 x 4 identity matrix
eye(4)

help eye
```

# Moving Data Around
```
A = [1,2; 3,4; 5,6]

% size of matrix_type
size(A)

% size of first dimension i.e. no of rows in A
size(A, 1)

% size of second dimension i.e. no of columns in A
size(A, 2)

v = [1, 2, 3, 4]

% size of the longest dimension
length(v)

% current location
pwd

% load data from file
load featuresX.dat
load priceY.dat

% variables in the memory currently
who

featuresX
priceY

size(featuresX)
size(priceY)

% detailed view of variables
whos

% To clear variables
clear featuresX
whos

% Saving data
v = priceY(1:10)
save hello.dat v

% clear all the variables
clear
whos

load hello.dat

% Save data in human readable format
save hello.txt v -ascii

% Manupulate Data
A = [1,2; 3,4; 5,6]
A(3,2)

A(2, :)
A(:, 2)

% Get everything from first and third rows
A([1,3], :)

% Appends another column vector to the right
A(:, 2) = [10; 11; 12]


A = [A, [100; 101; 102]]

% Put all elements of A into a single column vector
A(:)


A = [1,2; 3,4; 5,6]
B = [11,12; 13,14; 15,16]
C = [A B]
C = [A;B]
```

# Computing on Data
```
A = [1,2; 3,4; 5,6]

B = [11,12; 13,14; 15,16]

C = [1 1; 2 2]

A * C

% Multiply element with element at same position
A .* B

% Elementwise squaring
A .^ 2

v = [1; 2; 3]

% Elementwise reciprocal
1 ./ v
1 ./ A

% Elementwise logrithm
log(v)

% Elementwise exponentiation
exp(v)

% Elementwise absolute value
abs([-1; 2; -3])

-v

% Increamenting all elements by one
v + ones(length(v), 1)

% Another simple way to achieve above is
v + 1

% Transpose
A'
(A')'

a = [1 15 2 0.5]

% Max value in matrix a
val = max(a)

% Max value & it's index in matrix a
[val, ind] = max(a)

% Elementwise comparison operation
a < 3

% Find elements less than 3
find(a < 3)

max(A)

% Magic squares: all rows, columns & diagonals sum up to the same thing.
A = magic(3)

% Find row and columns greater than equal to 7. In this case (1,1), (3,2), (2,3)
[r, c] = find(A >= 7)

% add all elements of a
sum(a)

% Multiply all elements of a
prod(a)

% Round down all elements of a
floor(a)

% Round up all elements
ceil(a)

% Elementwise max of two randomly generated matrices
max(rand(3), rand(3))

% Take columnwise maximum. 1 below signifies takes max along the dimension 1.
max(A,[],1)

% Take rowwise maximum.
max(A,[],2)

% Default is columnwise max
max(A)

% To get the max element in entire matrix A
max(max(A))

% Or convert A to vector and then take max
max(A(:))

A = magic(9)

% Columnwise sum
sum(A, 1)

% Rowwise sum
sum(A, 2)

% Diagonalwise sum
sum(sum(A .* eye(9)))

% Sum along the other diagonals
sum(sum(A .* flipud(eye(9))))

A = magic(3)
temp = pinv(A)
temp*A
```

# Plotting Data
```
t = [0:0.01:0.98]
y1 = sin(2*pi*4*t)
plot(t,y1)

% Replaces sin plot above
y2 = cos(2*pi*4*t)
plot(t,y2)

% To plot both together
plot(t, y1)
hold on
plot(t, y2, 'r')
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('My Plot')

print -dpng 'myPlot.png'

figure(1); plot(t, y1);
figure(2); plot(t, y2);

% Divide a plot into 1 by 2 grid (first two parameters) and access first element (last parameter)
subplot(1,2,1);
plot(t, y1)
subplot(1,2,2)
plot(t, y2, 'r')

% set x range and y range. X axis from 0.5 to 1 and Y axis from -1 to 1.
axis([0.5 1 -1 1])

% Clear a figure
clf

A = magic(5)

% To visualize a matrix
imagesc(A)

% Comma chaining of function calls
imagesc(A), colorbar, colormap gray
```

# Control Statements: for, while, if statement
```
v = zeros(10, 1)

for i = 1:10
  v(i) = 2^i
end

indices = 1:10
for i = indices
  disp(i)
end

i = 1
while i <= 5
  v(i) = 100
  i = i+1
end

i = 1
while true
  v(i) = 999
  if i == 6
    break
  end
  i = i+1
end

v(1)
v(1) = 2
if v(1) == 1
  disp('The value is 1')
elseif v(1) == 2
  disp('The value is 2')
else
  disp('The value is not 1 or 2')
end

% Is the method is defined is some other path use addpath('C:\Users\bob\Desktop')
squareThisNumber(5)

[s, c] = squareAndCubeThisNumber(7)


X = [1 1; 1 2; 1 3];
Y = [1; 2; 3];
theta = [0; 1];

% Cost function comes 0 as line fits perfectly
j = costFunctionJ(X, Y, theta)

theta = [0; 0];
costFunctionJ(X, Y, theta)
```

# Vectorization
Example:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1896746a-de0f-475f-aceb-3bf4968fe93d)

Think of it as computing inner product of these two vectors. Where theta and X are as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/ed9dca94-f080-48db-b6f6-de39c868c183)

These two views can give us two different implementations.

## Unvectorized Implementation
```
prediction = 0.0
for j = 1:n+1,
  prediction = prediction + theta(j) * x(j)
end
```

## Vectorized Implementation
This implementation is simpler and much more efficient.
```
prediction = theta' * x
```

## Gradient Descent:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/c77fedaa-de24-4d98-9dac-b260d7f9d0cf)

Below are update rules for θ<sub>0</sub>, θ<sub>1</sub> & θ<sub>2</sub>

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/5651b86e-c6e3-4c2e-9851-d6bb1a4d5c64)

prediction = theta' * x

## Vectorized Implementation
θ = θ - α * δ

Where δ is n+1 dimensional vector:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/023f3d71-3b5e-46c6-98ff-b19b7f0460c2)

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e57aeaf5-4354-481c-b163-5ad87ae5258b)


