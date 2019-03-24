// Load iris data and remove class attribute
data = (new weka.core.converters.ConverterUtils.DataSource("/Users/tiffanyxu/Desktop/wekaFiles/newAdult.arff")).getDataSet()
data.deleteAttributeAt(0)

// Standardise the data to compare with R's output at http://www.sthda.com/english/wiki/print.php?id=239 (Section 5.2.3)
standardize = new weka.filters.unsupervised.attribute.Standardize()
standardize.setInputFormat(data)
data = weka.filters.Filter.useFilter(data, standardize)

// Specify the distance function to use
distance = new weka.core.EuclideanDistance()
distance.setDontNormalize(true) // Turn normalisation off because we have standardised the data
for (t = 30; t <= 100; t+=10) {
    // Specify the clustering algorithm
    clusterer = new weka.clusterers.SimpleKMeans()
    clusterer.setNumClusters(t) // Setting the number of clusters to 3 here as an example
    clusterer.setDistanceFunction(distance) // In k-means, we can use the same distance function
    clusterer.buildClusterer(data)
    
    // Find cluster index of each instance
    clusterIndexOfInstance = new int[data.numInstances()]
    for (i = 0; i < data.numInstances(); i++) {
      clusterIndexOfInstance[i] = clusterer.clusterInstance(data.instance(i))
    } 
    
    sumSilhouetteCoefficients = 0
    for (i = 0; i < data.numInstances(); i++) {
      
      // Compute average distance of current instance to each cluster, including its own cluster
      averageDistancePerCluster = new double[clusterer.numberOfClusters()]
      numberOfInstancesPerCluster = new int[clusterer.numberOfClusters()]
      for (j = 0; j < data.numInstances(); j++) {
        averageDistancePerCluster[clusterIndexOfInstance[j]] += distance.distance(data.instance(i), data.instance(j))
        numberOfInstancesPerCluster[clusterIndexOfInstance[j]]++ // Should the current instance be skipped though?
      }
      for (k = 0; k < averageDistancePerCluster.length; k++) {
        averageDistancePerCluster[k] /= numberOfInstancesPerCluster[k]
      }
     
      // Average distance to instance's own cluster
      a =  averageDistancePerCluster[clusterIndexOfInstance[i]]
      
      // Find the distance of the "closest" other cluster
      averageDistancePerCluster[clusterIndexOfInstance[i]] = Double.MAX_VALUE
      b = Arrays.stream(averageDistancePerCluster).min().getAsDouble()
    
      // Compute silhouette coefficient for current instance
      sumSilhouetteCoefficients += clusterer.numberOfClusters() > 1 ? (b - a) / Math.max(a, b) : 0
    }
    
    println("Average silhouette coefficient for " + t + ": " + (sumSilhouetteCoefficients / data.numInstances()))
}