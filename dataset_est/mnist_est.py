import numpy as np
from sklearn.datasets import fetch_openml
import tensorflow as tf
from tensorflow.keras.datasets import mnist



# Load MNIST dataset
def entropy_mnist():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist['data']
    y = mnist['target']

    # Calculate probability distribution
    counts = np.bincount(y.astype(int))
    probabilities = counts / len(y)

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    # print("Entropy: ", entropy)
    return entropy   



def mnist_distribuiton(eta: float):
    """
    This function is used for the distribution calculation of the MNIST dataset,
    eta -  sampling ratio is passed and datasets is split according to that and 
    then calculate the dataset distribution
    Args : 
        - eta - sampling ratio
    Return:
        - Probability Distribution of the datasets
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_labels = np.concatenate((train_labels, test_labels))
    split_val = len(all_labels)*eta
    split_labels = all_labels[:round(split_val)]    
    
    ## counting the how many samples are there in each class 
    class_counts = np.bincount(split_labels)
    
    total_sample = len(split_labels)
    class_distribution = class_counts / total_sample
    
    return class_distribution   ## this gives an array 
    
    

def kl_div_mnist(eta: float):
    """
    This used to calculate the KL divergence between the distributions of the sampled data 
    and the original datasets
    Args:
        - eta - sampling ratio
    Return 
        - KL-Div beteween the sampled data and the original dataset
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_labels = np.concatenate((train_labels, test_labels))
    
    class_counts = np.bincount(all_labels)
    
    total_sample = len(all_labels)
    class_distribution = class_counts / total_sample
    
    kl = np.sum(mnist_distribuiton(eta) * np.log(mnist_distribuiton(eta) / class_distribution)) 
    
    return kl
