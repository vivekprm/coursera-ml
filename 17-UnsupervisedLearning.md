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

# K-means Optimization Objective
While the k means is running we are keeping track of following variables:
- c<sup>(i)</sup> = index of cluster (1,2,...,K) to which example x<sup>(i)</sup> is currently assigned.
- µ<sub>k</sub> = cluster centroid k (µ<sub>k</sub> belongs to R<sup>n</sup>)

For k-means we use K to denote the total number of Clusters and here k is going to be an index into the cluster centroids k belongs to {1, 2, 3, ..., K}
Here is one more notation, which is gonna use:
µ<sub>c<sup>(i)</sup></sub> = cluster centroid of cluster to which example x<sup>(i)</sup> has been assigned. E.g. let's say x<sup>(i)</sup> -> 5, hence c<sup>(i)</sup> = 5 and hence µ<sub>c<sup>(i)</sup></sub>=µ<sub>5</sub>.

Below is our optimisation objective:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/118a5f3d-aea0-49cd-8798-3b09f261655f)

This Cost Function is sometimes also called Distortion.

To provide more details:
First step in the above K-means algorithm is Cluster Assignment Step, where we assign each point to Cluster Centroid and it's possible to show mathematically that what the Cluster Assignment step is doing is exactly minimizing J(...) with respect to variables c<sup>(1)</sup>, c<sup>(2)</sup>, ...., c<sup>(m)</sup> while holding the Cluster Centroid µ<sub>1</sub>, ..., µ<sub>k</sub> fixed.

So first step doesn't change the cluster centroid but what it's doing is it's picking the values of c<sup>(1)</sup>, c<sup>(2)</sup>, ...., c<sup>(m)</sup> that minimizes the cost function J.

Second step in the above k-means algorithm is Move Centroid Step and it can be shown mathematically that it chooses the values of µ that minimizes J wrt locations of the cluster centroids µ<sub>1</sub>, ..., µ<sub>k</sub>

# Random Initialization
We are going to see how we can optimize k-means to avoid local optima. 

One step that we didn't talk about in k-means algorihtm is the first step which is:
Randomly initialize K cluster centroids µ<sub>1</sub>, ..., µ<sub>k</sub> belongs to R<sup>n</sup>

There are few different ways that one can imagine using to randomly initialize the Cluster Centroid. But it turns out there is one method that is much more recommended than most of the other options.

Here is how we initialize Cluster Centroid:
- Should have K < m
- Randomly pick K training examples.
- Set µ<sub>1</sub>, ..., µ<sub>k</sub> equal to these K examples.

E.g. Let's say K = 2 so in below example we want to find out two Clusters:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/021a3d23-eac4-4715-9217-db4c72917268)

- So we randomly pick two examples depicted with an arrow and initialize our two Cluster Centroid with these.

Here in this case we got lucky but we might have picked centroids as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/2d4a0008-8b16-4038-85b1-bf6380a7574e)

So that's how we can randomly initialize the Cluster Centroids.

Looking at these two illustrations, you might guess that K-means can endup converging to different solutions depending on exactly how the Clusters were initialized and so depending upon the random initialization K-means can endup at different solutions and in particular K-means can actually endup at local optima.

If we have given a data set as below it looks like there are three clusters and if you run k-means and if it ends up at a good local optima you may end up with cluster looking as below:

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/aa442ffc-df1a-4930-90ad-417f91dad42a)

But if you are unlucky in random initialisation k-means can also get stuck at different local optima.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/45217fd7-f1a7-4e5f-800e-14faa09f1cb5)

So in the left example it seems blue cluster has captured lots of examples and red and green are capturing relatively small number of points. Similar is the case in the right.

So if you are worried k-means getting stuck in local optima, if you want to increase the odds of k-means finding the best possible clustering, like shown in the above pic, what we can do is, try multiple random initialisations. So instead of just initialising k-means once and hoping that it works, what we can do is, initialise k-means lots of times and use that to try to make sure we get as good a solution, as good a local or global optima as possible. Here is how you can go about doing that:

Let's say we decided to run k-means 100 times.

for i = 1 to 100 {
  Randomly initialize K-means.
  Run K-means. Get c<sup>(1)</sup>, c<sup>(2)</sup>, ...., c<sup>(m)</sup>, µ<sub>1</sub>, ..., µ<sub>k</sub>
  Compute Cost Function (distortion) J(c<sup>(1)</sup>, ...., c<sup>(m)</sup>, µ<sub>1</sub>, ..., µ<sub>k</sub>)
}

Pick clustering that gave lowest cost J(c<sup>(1)</sup>, ...., c<sup>(m)</sup>, µ<sub>1</sub>, ..., µ<sub>k</sub>).

If K is very large may be 100, so many random initializations may not make a huge difference and there is much higher chance that your first random initialization will give you pretty decent solution already and doing multiple random initializations will probably give slightly better solution but may be not that much.

But if you have small number of clusters may be 2-10 then random initialization may make huge difference.
