# Matrices and Vectors

Matrices are 2-dimensional arrays:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/05b74d5a-0beb-4aed-b230-a47fc72dad5d)

The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:
[w
x
y
z]

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.

# Notation and terms
- A<sub>ij</sub> refers to the element in the i<sup>th</sup> row and j<sup>th</sup> column of matrix A.
- A vector with 'n' rows is referred to as an 'n'-dimensional vector.
- v<sub>i</sub> refers to the element in the ith row of the vector.
- In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
- Matrices are usually denoted by uppercase names while vectors are lowercase.
- "Scalar" means that an object is a single value, not a vector or matrix.
- R refers to the set of scalar real numbers.
- R<sup>n</sup> refers to the set of n-dimensional vectors of real numbers.

Run the cell below to get familiar with the commands in Octave/Matlab. Feel free to create matrices and vectors and try out different things.
```
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```

# Addition & Scalar Multiplication
```
% Initialize matrix A and B and C 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s
```

# Matrix Vector Multiplication
Instead of going through loops (can be very high) and computing the multiplication we can use below trick of Matrix multiplication. Which is more concise and efficient.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1154fe64-2f3d-441e-941e-67e99f8f4b5d)

```
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9] 

% Initialize vector v 
v = [1; 1; 1] 

% Multiply A * v
Av = A * v
```

# Matrix Matrix Multiplication
![image](https://github.com/vivekprm/coursera-ml/assets/2403660/e60e2f44-9d2c-4e7d-a11d-f2188546367e)

You can see with one Matrix multiplication we are able to make 12 predictions.
```
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B

% Make sure you understand why we got that result
```

# Matrix Multiplication Properties
Matrices are not commutative: A x B != B x A
Matrices are associative: (A x B) x C = A x (B x C)

The identity matrix, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity 
matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

When multiplying the identity matrix after some matrix (A x I), the square identity matrix's dimension should match the other matrix's columns. When multiplying 
the identity matrix before some other matrix (I x A), the square identity matrix's dimension should match the other matrix's rows.
```
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA
```

# Inverse & Transpose
The inverse of a matrix A is denoted A<sup>-1</sup>. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the pinv(A) function and in Matlab with the inv(A) function. 
Matrices that don't have an inverse are **singular** or **degenerate**.

The transposition of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in matlab 
with the transpose(A) function or A':

```
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = inv(A)

% What is A^(-1)*A? 
A_invA = inv(A)*A
```
