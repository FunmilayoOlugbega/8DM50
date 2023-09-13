def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def mean_squared_error(X, y, beta):
    """
    Calculate the mean squared error of the model
    :param X: Input data matrix
    :param y: Target vector
    :beta: Estimated coefficient vector for the linear regression
    :return: Mean squared error
    """
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    mse= np.mean(((y-np.dot(X,beta)))**2)

    return mse

def normalize_data(Data_array):
    """
    This method will normalise array Data_array
    
    Parameter(s)
    -----------
    Data_array: array
        An array with data points which will need to be normalised
        
    Return(s)
    ----------
    X_normalized: array
        An array containing the normalised values of array Data_array
    """    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(Data_array.data)

    return X_normalized

def generate_sets(Data_array, Size_trainset = 300, Perc_validationset = 0.8):
    """
    This method will use normalize_data to normalise the data in Data_array.
    Besides it will splitt the data into the training, validation and test sets
    
    Parameter(s)
    -----------
    Data_array: array
        An array with data points which will need to be normalised and split 
    Size_trainset: int
        An intiger defining the size of the trainingset. The trainingset will contain Size_trainingset rows
    Data_array: float
        A float defining the size of the validation set. After splitting the trainingset, the other rows will be divided in a validation and test set. 
        Data_arrat is the percentage that will end up in the validation set.
        
    Return(s)
    ----------
    X_train_cancer: array
        An array containing the normalised values of array Data_array
    X_validate_cancer
        An array containing the normalised values of array Data_array
    X_test_cancer
        An array containing the normalised values of array Data_array
    y_train_cancer
        An array containing the tagerts of X_train_cancer
    y_validate_cancer
        An array containing the tagerts of X_validate_cancer
    y_test_cancer
        An array containing the tagerts of X_test_cancer
    """
    # normalise dataset
    Data = Data_array.data
    X_normalized = normalize_data(Data_array)
        
    # Load targets 
    y_train = Data_array.target[:Size_trainset]
    y_interim = Data_array.target[Size_trainset:]
    y_validate = y_interim[:int(Perc_validationset*len(y_interim))]
    y_test = y_interim[int(Perc_validationset*len(y_interim)):]
    
    # Split the data into training and testing sets
    # The first 300 patients will be used as training set
    X_train = X_normalized[:Size_trainset]
    X_interim = X_normalized[Size_trainset:]
    X_validate = X_interim[:int(Perc_validationset*len(X_interim))]
    X_test = X_interim[int(Perc_validationset*len(X_interim)):]

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def Euclidean_distance(x1, x2):
    """
    This method calculates the euclidean distance between two points x1 and x2
    
    Parameter(s)
    -----------
    x1: array
    x2: array
        
    Return(s)
    ----------
    eucl_dist: int
        the euclidian distance between x1 and x2
    """
    eucl_dist=np.sqrt(np.sum((x1 - x2) ** 2))
    return eucl_dist

def predict_classification(X_train, y_train, X_validate, k=5):
    """
    This method is a k-NN clasifier, which predict the label of the test points 
    based on the k-nearest neigbors.
    
    Parameter(s)
    -----------
    X_train: array
        An array containing the normalised values of all features of the trainingset
    y_train: array
        An array containing the tagerts of X_train
    X_validate: array
        An array containing the normalised values  of all features of the validationset
    k: int
        number of neighbours is used to predict the label of the target class of the point
    
    Return(s)
    np.array(y_pred): int
        the predicted value for the points of the validation set

    """
    y_pred = []
    
    for x in X_validate:
        
        # Compute distances between x and all examples in the training set
        distances = [Euclidean_distance(x, x_train_single) for x_train_single in X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_neighbors_indices = np.argsort(distances)[:k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [y_train[i] for i in k_neighbors_indices]
        
        # Return the most common class label among the k neighbors
        most_common = np.bincount(k_neighbor_labels).argmax()
        y_pred.append(most_common)
    
    return np.array(y_pred)

def Evaluate(y_true, y_pred):
    """
    This is method determines the accuracy of the k-NN classifier based on
    true labels and the predicted labels.
    
    Parameter(s)
    y_true: array
        An array of the true labels of the points
    y_pred: array
        An array of the predicted labels of the points
        
    Return(s)
    acc: int
        Accuracy of the k-NN clasifier
    """
    n_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            n_correct += 1
    if len(y_true) != 0:
        acc = n_correct/len(y_true)
    else:
        acc = 0
    return acc

def predict_regression(X_train, y_train, X_test, k=5):
    """
    This method is a k-NN regressor, which predict the average target value 
    of the test points based on the k-nearest neigbors.
    
    Parameter(s)
    -----------
    X_train: array
        An array containing the normalised values of all features of the trainingset
    y_train: array
        An array containing the tagerts of X_train
    X_validate: array
        An array containing the normalised values  of all features of the validationset
    k: int
        number of neighbours is used to predict the label of the target class of the point
    
    Return(s)
    np.array(y_pred): int
        the predicted value for the points of the validation set

    """
    y_pred = []  
    for x in X_test:
        # Compute distances between x and all examples in the training set
        distances = [Euclidean_distance(x, x_train) for x_train in X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_neighbors_indices = np.argsort(distances)[:k]
        
        #calculate average  distance
        k_neighbor_values = [y_train[i] for i in k_neighbors_indices]
        predicted_value = np.mean(k_neighbor_values)
        y_pred.append(predicted_value)

    return np.array(y_pred)

def mse_score(y_true, y_pred):
    """
    Method to determine the mean squared error of the k-NN regressor
    
    Parameter(s)
    -----------
    y_true: array
        An array of the true labels of the points
    y_pred: array
        An array of the predicted labels of the points
        
    Return(s)
    ----------
    mse: int
        The Mean Squared Error
    """
    mse=np.square(np.subtract(y_true,y_pred)).mean()
    return mse

def mean_and_std(data):
    means_list = []
    stds_list = []
    
    for x in range(data.shape[1]):
        mean = np.mean(data[:, x])
        std = np.std(data[:, x])
        means_list.append(mean)
        stds_list.append(std)
    means = np.array(means_list)
    stds = np.array(stds_list)
    
    return means, stds