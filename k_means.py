import numpy as np

class KMeans(object):
  """ a K-means clustering with L2 distance """

  def __init__(self, num_clusters=3):
    self.num_clusters = num_clusters

  def train(self, X, epsilon=1e-12):
    """
    Train the k-means clustering.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - epsilon: (float) lower limit to stop cluster.
    """
    self.centroids = X[np.random.choice(np.arange(X.shape[0]),size=self.num_clusters)]

    # save the change of centroids position after updating
    change = np.ones(self.num_clusters)

    while (all(num > epsilon for num in change)):
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all cluster       #
        # centroid then assign them to the nearest centroid.                    #
        #########################################################################
        cluster = self.predict(X)
        #########################################################################
        # TODO:                                                                 #
        # After assigning data to the nearest centroid, recompute the centroids #
        # then calculate the differrent between old centroids and the new one   #
        #########################################################################
        new_centroids = np.zeros(self.centroids.shape)
        for i  in range(new_centroids.shape[0]):
              new_centroids[i] = np.mean(X[cluster==i],axis=0)
              change[i]  = np.sqrt(np.sum((new_centroids[i] -self.centroids[i]) **2 , axis=-1) ) 
        self.centroids = new_centroids
        # break
        
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        
  def predict(self, X, num_loops=0):
    """
    Predict labels for test data using this clustering.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - num_loops: Determines which implementation to use to compute distances
      between cluster centroids and testing points.
    Returns:
    - y: (A numpy array of shape (num_test,) containing predicted clusters for the
      test data, where y[i] is the predicted clusters for the test point X[i]).  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value {} for num_loops'.format(num_loops))

    return self.predict_clusters(dists)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each cluster centroid point
    in self.centroids using a nested loop over both the cluster centroids and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_clusters) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth cluster centroid.
    """
    num_test = X.shape[0]
    dists = np.zeros((num_test, self.num_clusters))
    for i in range(num_test):
      for j in range(self.num_clusters):
            dists[i][j] = np.sqrt(np.sum((X[i] - self.centroids[j])**2))
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # cluster centroid, and store the result in dists[i, j]. You should #
        # not use a loop over dimension.                                    #
        #####################################################################
        # pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each cluster centroid
    in self.centroids using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    dists = np.zeros((num_test, self.num_clusters))
    for i in range(num_test):
        dists[i] = np.sqrt(np.sum((X[i] - self.centroids)**2,1))

      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all cluster  #
      # centroids, and store the result in dists[i, :].                     #
      #######################################################################
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each cluster centroid
    in self.centroids using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    dists = np.zeros((num_test, self.num_clusters)) 
    dists = np.sqrt(np.sum((np.expand_dims(X,axis=1) -self.centroids) **2 , axis=-1) )    
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all cluster       #
    # centroid without using any explicit loops, and store the result in    #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_clusters(self, dists):
    """
    Given a matrix of distances between test points and cluster centroids,
    predict a cluster for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_clusters) where dists[i, j]
      gives the distance betwen the ith test point and the jth cluster centroid.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted cluster for the
      test data, where y[i] is the predicted cluster for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the nearest cluster of the ith        #
      # testing point.                                                        #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      nearest = np.argsort(dists[i])[0]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = nearest
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

