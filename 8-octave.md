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
