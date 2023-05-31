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
