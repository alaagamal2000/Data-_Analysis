import numpy as np
#  creating a rank 2 ndarray of random integers between 0 and 5,000 (inclusive) with 1000 rows and 20 columns.
a=np.random.randint(0,5001,size=(1000,20))
print("shape of a ",a.shape)
#Norm_Col𝑖=coli−𝜇𝑖/𝜎𝑖
#here  Col𝑖  is the  𝑖 th column of  𝑋 ,  𝜇𝑖  is average of the values in the  𝑖 th column of  𝑋 ,
# and  𝜎𝑖  is the standard deviation of the values in the  𝑖 th column of  𝑋 .
# In other words, mean normalization is performed by subtracting from each column of  𝑋  the average of its values,
# and then by dividing by the standard deviation of its values.
## Average of the values in each column of a
meanofa=np.mean(a,axis=0)
# Standard Deviation of the values in each column of a
stdofa=np.std(a,axis=0)
# Print the shape of ave_cols
print(meanofa.shape)

# Print the shape of std_cols
print(stdofa.shape)
# calculate Norm_col of a
Norm_a=a-meanofa/stdofa
print("Norm_a=",Norm_a)
#If you have performed the mean normalization correctly,
# then the average of all the elements in  𝑋norm  should be close to zero,
# and they should be evenly distributed in some small interval around zero. You can verify this by filing the code below:
# Print the average of all the values of X_norm
print("mean of Norm_a",Norm_a.mean())

# Print the average of the minimum value in each column of X_norm

Norm_a.min(axis=0).mean()
# Print the average of the maximum value in each column of X_norm
Norm_a.max(axis=0).mean()
# Create a rank 1 ndarray that contains a random permutation of the row indices of `X_norm`
row_a = np.random.permutation(Norm_a.shape[0])
print(row_a)
#----------------------------------needed to be understand---------------------------------------------------------
# Let's get the count of 60% rows. Since, len(X_norm) has a lenght 1000, therefore, 60% = 600
sixty = int(len(Norm_a) * 0.6)

# Let's get the count of 80% rows
eighty = int(len(Norm_a) * 0.8)

# Create a Training Set
# Here row_indices[:sixty] will give you first 600 values, e.g., [93 255 976 505 281 292 977,.....]
# Those 600 values will will be random, because row_indices is a 1-D array of random integers.
# Next, extract all rows represented by these 600 indices, as X_norm[row_indices[:sixty], :]
X_train = Norm_a[row_a[:sixty], :]

# Create a Cross Validation Set
X_crossVal = Norm_a[row_a[sixty: eighty], :]

# Create a Test Set
X_test = Norm_a[row_a[eighty: ], :]
#If you performed the above calculations correctly,
# then X_tain should have 600 rows and 20 columns, X_crossVal should have 200 rows and 20 columns,
# and X_test should have 200 rows and 20 columns. You can verify this by filling the code below:

# Print the shape of X_train
print(X_train.shape)
# Print the shape of X_crossVal
print(X_crossVal.shape)
# Print the shape of X_test
print(X_test.shape)

