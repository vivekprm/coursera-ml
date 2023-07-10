# Clustering
Here is the typical Supervised Learning Problem, where we are given a labeled training set and the goal is to find the decision boundary that separates the positive label examples and the negative label examples. 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/3a6d2640-d37f-4df8-8651-d157af5140f8)

So, the Supervised Learning problem in this case is given a set of labels to fit a hypothesis to it.

In contrast, in the Unsupervised Learning, we have data that doesn't have any labels associated with it. So we are given data that looks as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/f4541301-6bf6-447b-a10a-6c325e1ae14d)

Here is a set of points with no labels and so our training set written just {x<sup>(1)</sup>, x<sup>(2)</sup>, x<sup>(3)</sup>, ...., x<sup>(m)</sup>} and we don't get any labels y and that's why the points plotted up on the figure don't have any labels them.

So, in Unsupervised Learning what we do is we give this sort of unlabeled training set to an algorithm and we just ask the algorithm find some strucutre in the data for us. E.g. in above dataset we might have points grouped into two separate clusters.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/d4a54051-fd87-48be-95ac-09c98dabd2cb)

So an algorithm that finds the clusters like the ones I've just circled is called a Clustering Algorithm. This will be our first type of Unsupervised Learning.

## K-Means
It's by far the most widely used clustering algorithm.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/4996ea62-18d8-42df-ae62-3f0e311f5657)

Let's say we want to take unlabeled dataset like above and we want to group the data into two clusters.

If we run the K-Means clustering algorithm, here is what we  are going to do. The first step is to randomly initialize two points (because we want to group into two clusters) called, the cluster centroids. 

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/82920f47-1ab4-4778-b8c4-ea9dac6a6717)

K-Means is an iterative algorithm and it does two things: 
- First is a Cluster Assignment step. 
  - It's going to go through each of the examples and depending upon whether it's closer to the red cluster centroid or the blue cluster centroid it's going to assign each of the data points to one of the two cluster centroids. So it goes through dataset and color each of the points either red or blue depending upon whether it's closer to the red cluster centroid or the blue cluster centroid.
- And second is a Move Centroid step.
  - Take the two Cluster Centroids i.e. red cross and the blue cross and going to move them to the average of the points coloured the same colour.
- Then again we go to the step one and iterate again till the CLuster Centroids don't change and color of the points don't change further.

Let's write it more formally:
Input:
- K (number of Clusters we want to find)
- Training Set {x<sup>(1)</sup>, x<sup>(2)</sup>, x<sup>(3)</sup>, ...., x<sup>(m)</sup>}

For K-Means we are going to use convention that x<sup>(i)</sup> is R<sup>n</sup> dimensional vector. We are dropping x<sub>0</sub>=1

This is what the K-Means algorithm does:
- Randomly initialize K cluster centroids μ<sub>1</sub>, μ<sub>2</sub>, ...., μ<sub>k</sub> belong to R<sup>n</sup>
- Repeat {
    - for i = 1 to m
      - c<sup>(i)</sup> := index(from 1 to K) of cluster centroid closest to x<sup>(i)</sup>
    - for k = 1 to K
      -  μ<sub>k</sub> := average (mean) of points assigned to cluster k
- }

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2bcb1897-a808-4638-b483-a9493e8f2684)

## K-mean for non-separated Clusters
So far we looked at dataset that had well separated clusters but it turns out that very often K means is also applied to datasets that may not have very well separated clusters. Here is an example application of t-shirt sizing.

Let's say you are a t-shirt manufacturer and what you've done is you've gone to the population that you want to sell t-shirts to and you've collected a number of examples of height and weight of these people in your population and so may be endup with dataset like this:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/1ca46899-b118-4745-ae3e-813e14a989f2)

Now let's say you want to size your t-shirts in S, M, L. So how big should I make my small, medium and large ones?
One way to do this would be to run k-means clustering algorithm on this dataset and assign it two 3 different clusters.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/fac6a68e-e379-4646-a8b9-379c952b41a0)

This is similar to market segmentation problem.
