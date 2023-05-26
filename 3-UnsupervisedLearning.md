In this category we are given a dataset and our purpose is to find a structure (cluster), unlike in supervised learning where we were given cancer type 
malignant & benign.

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/48712534-4adb-4c22-be5a-8ad2ea542f71)

One example where clustering is used is google news. Google news crawls and groups cohesive sub stories.

# Clustering
Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different 
variables, such as lifespan, location, roles, and so on.

## Examples
- Organize Computing Clusters
  - Figure out which machines work together from a large data center.  
  - If you put them together your datacenter works more efficiently.
- Social Network Analysis
  - Given your facebook friends or google plus circle, can we automatically identify, which are cohesive group of friends.
- Market Segmentation
  - Look at customer dataset and automatically discover market segments and automatically group your customers into different market segments.
- Astronomical Data Analysis
  - How galaxies are formed.

# Non-clustering
The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at 
a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)).

![image](https://github.com/vivekprm/coursera-ml/assets/2403660/be9dde81-7fbc-40d8-b22e-91a90f8aae16)

Take the recording of these two microsphones and give it to Unsupervised Learning ALgorithm and tell the algorithm find structure in this data and should be able
to separate out different audio voices.

Seems it should require lots of audio processing etc. However, it turns out just require 1 line of code.
```octave
[W,s,v] = svd((repmat(sum(x.*x,1), size(x,1),1).*x)*x');
```
