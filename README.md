#Part 1: Iris Clustering

Description:

This is the famous Iris dataset and serves as an easy benchmark for evaluation. Test your K-Means Algorithm on this easy dataset with 4 features:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
and 150 instances.
Essentially assign the 150 instances in the test file to 3 cluster ids given by 1, 2 or 3.
The file iris_new_data.txt, under "Test data," contains the data you use for clustering.

#Part 2: Image Clustering

Description: 

The objectives of this assignment are the following: 
Implement the K-Means Algorithm
Deal with Image data (processed and stored in vector format) Think about Best Metrics for Evaluating Clustering Solutions

Detailed Description:
For this assignment, you are required to implement the K-Means algorithm. Please do not use libraries for this assignment except for pre-processing.
Input Data (provided as new_test.txt) consists of 10,000 images of handwritten digits (0-9). The images were scanned and scaled into 28x28 pixels. For every digit, each pixel can be represented as an integer in the range [0, 255] where 0 corresponds to the pixel being completely white, and 255 corresponds to the pixel being completely black. This gives us a 28x28 matrix of integers for each digit. We can then flatten each matrix into a 1x784 vector. No labels are provided.
Format of the input data: Each row is a record (image), which contains 784 comma-delimited integers.
Essentially your task is to assign each of the instances in the input data to K clusters identified from 1 to K. 
