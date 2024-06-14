import pickle

from svc import SVC
from datamodule import MNIST, CIFAR10

#########################################################################################
# Hyperparameters: (Remember to modify these in svm.py and mksvm.py when reproduction.)
# - mnist
#     - linear (penalty=3)
#     - gaussian (penalty=3, gamma=0.01)
#     - polynomial (penalty=3, gamma=0.03, bias=1, d=3)
#     - sigmoid (penalty=3, gamma=0.007, bias=-1)
# - cifar10
#     - linear (penalty=3)
#     - gaussian (penalty=3, gamma=0.008)
#     - polynomial (penalty=3, gamma=0.02, bias=1, d=3)
#     - sigmoid (penalty=3, gamma=0.001, bias=-1)
# - cifar10-features
#     - linear (penalty=3)
#     - gaussian (penalty=3, gamma=0.05)
#     - polynomial (penalty=3, gamma=0.04, bias=1, d=3)
#     - sigmoid (penalty=3, gamma=0.01, bias=-1)
##########################################################################################


if __name__ == '__main__':
    DATASET = 'cifar10'  # 'mnist', 'cifar10', 'cifar10-features'

    if DATASET == 'mnist':
        X_train, y_train = MNIST('./dataset/mnist_data/', group='train')
        X_test, y_test = MNIST('./dataset/mnist_data/', group='test')
        X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
    elif DATASET == 'cifar10':
        X_train, y_train = CIFAR10('./dataset/cifar-10-batches-py/', group='train')
        X_test, y_test = CIFAR10('./dataset/cifar-10-batches-py/', group='test')
        X_train, X_test = X_train.reshape(-1, 3 * 32 * 32), X_test.reshape(-1, 3 * 32 * 32)
    else:
        with open('train_features.pkl', 'rb') as f:
            X_train_features, y_train = pickle.load(f)
            X_train = X_train_features.reshape(-1, 7 * 7)
        with open('test_features.pkl', 'rb') as f:
            X_test_features, y_test = pickle.load(f)
            X_test = X_test_features.reshape(-1, 7 * 7)


    # Train!
    print("Dataset: ", DATASET, '\n')
    valid_kernel_combinations = ['linear', 'gaussian', 'polynomial', 'sigmoid',
                                 ['linear', 'gaussian'], ['linear', 'polynomial'], ['linear', 'sigmoid'],
                                 ['gaussian', 'polynomial'], ['gaussian', 'sigmoid'], ['polynomial', 'sigmoid'],
                                 ['linear', 'gaussian', 'polynomial'], ['linear', 'gaussian', 'sigmoid'],
                                 ['linear', 'polynomial', 'sigmoid'], ['gaussian', 'polynomial', 'sigmoid'],
                                 ['linear', 'gaussian', 'polynomial', 'sigmoid']]

    for i in range(len(valid_kernel_combinations)):
        print("Current Kernel Combination: ", valid_kernel_combinations[i])
        print("Current Kernel Combination Index: ", i + 1)
        print("Start Training!")
        if i < 4:
            model = SVC(kernel=valid_kernel_combinations[i], strategy='OvsO', multi_kernel=False)
        else:
            model = SVC(kernel=valid_kernel_combinations[i], strategy='OvsO', multi_kernel=True)
        model.fit(X_train, y_train)

        acc_train = model.score(X_train, y_train)
        print(f"Training accuracy: {acc_train}")
        acc_test = model.score(X_test, y_test)
        print(f"Test accuracy: {acc_test}")
        print('\n')
