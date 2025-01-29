import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
 
 
def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.
    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy Imputing By User: {}".format(acc))
    return acc
 
 
def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
 
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T

    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy Imputing By Item: {}".format(acc))

    return acc
 
 
def main():
    sparse_matrix = load_train_sparse(r"ProjectDefault\data").toarray()
    val_data = load_valid_csv(r"ProjectDefault\data")
    test_data = load_public_test_csv(r"ProjectDefault\data")
 
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
 
    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    #impute by user
    k_values = [1, 6, 11, 16, 21, 26]
    val_acc = []
    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        val_acc.append(acc)
    plt.plot(k_values, val_acc)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs. k By User")
    plt.show()
    k_optimal = k_values[np.argmax(val_acc)]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_optimal)
    print("Test Accuracy Imputing By User: {}".format(test_acc))
 
    #impute by item
    k_values = [1, 6, 11, 16, 21, 26]
    val_acc = []
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc.append(acc)
    plt.plot(k_values, val_acc)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs. k By Item")
    plt.show()
    k_optimal = k_values[np.argmax(val_acc)]
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_optimal)
    print("Test Accuracy Imputing By Item: {}".format(test_acc))
 
if __name__ == "__main__":
    main()