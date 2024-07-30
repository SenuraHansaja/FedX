import numpy as np
import torchvision


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False,)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False,)
# Extract images and labels from the dataset
X_train = np.array([img for img, _ in trainset])
y_train = np.array([label for _, label in trainset])

X_test = np.array([img for img, _ in testset])
y_test =  np.array([label for _, label in testset])



def entropy_cifar10(X_train = X_train,
                    X_test = X_test,
                    y_train = y_train,
                    y_test = y_test):
    # Combine train and test sets
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    # Calculate probability distribution
    counts = np.bincount(y.flatten())
    probabilities = counts / len(y)

    # Calculate entropy
    entropy = -1*np.sum(probabilities * np.log2(probabilities))
    # print("Entropy: ", entropy)
    return entropy