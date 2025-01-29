from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    log_lklihood = 0.0

    for i in range(len(data["question_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]

        theta_i = theta[user_id]
        beta_j = beta[question_id]

        p = sigmoid(theta_i - beta_j)

        log_lklihood += c_ij * np.log(p) + (1 - c_ij) * np.log(1 - p)

    return -log_lklihood

def regular_neg_log_likelihood(data, theta, beta, lambda_theta, lambda_beta, reg_type):
    """Compute the REGULARIZED negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param lambda_theta: float
    :param lambda_beta: float
    :param reg_type: string         ("l1", "l2" or "elastic_net")
    :return: float
    """
    log_lklihood = 0.0

    for i in range(len(data["question_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]

        theta_i = theta[user_id]
        beta_j = beta[question_id]

        p = sigmoid(theta_i - beta_j)

        log_lklihood += c_ij * np.log(p) + (1 - c_ij) * np.log(1 - p)

    # regularization term
    if reg_type == "l1":
        reg = lambda_theta * np.sum(np.abs(theta)) + lambda_beta * np.sum(np.abs(beta))
    elif reg_type == "l2":
        reg = lambda_theta * np.sum(theta**2) * 0.5 + lambda_beta * np.sum(beta**2) * 0.5
    elif reg_type == "elastic_net":
        reg = (
            lambda_theta * np.sum(np.abs(theta))
            + lambda_beta * np.sum(np.abs(beta))
            + lambda_theta * np.sum(theta**2) * 0.5
            + lambda_beta * np.sum(beta**2) * 0.5
        )
    else:
        raise ValueError("Invalid regularization type")

    return -log_lklihood + reg


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)

    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]

        theta_i = theta[user_id]
        beta_j = beta[question_id]

        p = sigmoid(theta_i - beta_j)

        d_theta[user_id] += c_ij - p
        d_beta[question_id] += p - c_ij

    theta += lr * d_theta
    beta += lr * d_beta

    return theta, beta

def regular_update_theta_beta(data, lr, theta, beta, lambda_theta, lambda_beta, reg_type):
    """Update theta and beta using gradient descent with REGULARIZATION.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param lambda_theta: float
    :param lambda_beta: float
    :param reg_type: string         ("l1", "l2" or "elastic_net")
    :return: tuple of vectors
    """
    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)

    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]

        theta_i = theta[user_id]
        beta_j = beta[question_id]

        p = sigmoid(theta_i - beta_j)

        d_theta[user_id] += c_ij - p
        d_beta[question_id] += p - c_ij

    # update theta and beta with regularization
    if reg_type == "l1":
        theta += lr * d_theta - lambda_theta * np.sign(theta)
        beta += lr * d_beta - lambda_beta * np.sign(beta)
    elif reg_type == "l2":
        theta += lr * d_theta - lambda_theta * theta
        beta += lr * d_beta - lambda_beta * beta
    elif reg_type == "elastic_net":
        theta += lr * (d_theta - lambda_theta * np.sign(theta) - lambda_theta * theta)
        beta += lr * (d_beta - lambda_beta * np.sign(beta) - lambda_beta * beta)
    else:
        raise ValueError("Invalid regularization type")
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.randn(len(set(data["user_id"])))
    beta = np.random.randn(len(set(data["question_id"])))

    val_acc_lst = []
    train_log_likelihoods = []
    val_log_likelihoods = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta, beta)

        train_log_likelihoods.append(-neg_lld_train)
        val_log_likelihoods.append(-neg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, train_log_likelihoods, val_log_likelihoods

def regular_irt(data, val_data, lr, iterations, lambda_theta, lambda_beta, reg_type):
    """Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param lambda_theta: float
    :param lambda_beta: float
    :param reg_type: string         ("l1", "l2" or "elastic_net")
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.random.randn(len(set(data["user_id"])))
    beta = np.random.randn(len(set(data["question_id"])))

    val_acc_lst = []
    train_log_likelihoods = []
    val_log_likelihoods = []

    for i in range(iterations):
        neg_lld_train = regular_neg_log_likelihood(data, theta, beta, lambda_theta, lambda_beta, reg_type)
        neg_lld_val = regular_neg_log_likelihood(val_data, theta, beta, lambda_theta, lambda_beta, reg_type)

        train_log_likelihoods.append(-neg_lld_train)
        val_log_likelihoods.append(-neg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = regular_update_theta_beta(data, lr, theta, beta, lambda_theta, lambda_beta, reg_type)

    return theta, beta, val_acc_lst, train_log_likelihoods, val_log_likelihoods


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # hyperparameters
    learning_rate = 0.005
    iterations = 100

    # training
    theta, beta, val_acc_lst, train_log_likelihoods, val_log_likelihoods = irt(train_data, val_data, learning_rate, iterations)

    # plot training and validation log-likelihood vs interation
    plt.plot(range(1, iterations + 1), train_log_likelihoods, label="Training Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.title("Training Log-Likelihoods")
    plt.show()

    plt.plot(range(1, iterations + 1), val_log_likelihoods, label="Validation Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.title("Validation Log-Likelihoods")
    plt.show()

    # evaluation
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Final Validation Accuracy: {}".format(val_acc))
    print("Final Test Accuracy: {}".format(test_acc))

    j1, j2, j3 = 1, 2, 3

    # plot sigmoid function
    thetas = np.linspace(-3, 3, 100)
    plt.plot(thetas, sigmoid(thetas - beta[j1]), label=f"Question {j1}")
    plt.plot(thetas, sigmoid(thetas - beta[j2]), label=f"Question {j2}")
    plt.plot(thetas, sigmoid(thetas - beta[j3]), label=f"Question {j3}")
    plt.xlabel("Student Ability (θ)")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.title("Probability of Correct Response vs. Student")
    plt.show()

def part_b():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # hyperparameters
    learning_rate = 0.005  # unchanged
    iterations = 100  # unchanged
    lambda_beta = 0.005
    lambda_theta = 0.005

    # training without regularization
    theta, beta, val_acc_lst, train_log_likelihoods, val_log_likelihoods = irt(train_data, val_data, learning_rate, iterations)
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f"Final Validation Accuracy (Original): {val_acc}")
    print(f"Final Test Accuracy (Original): {test_acc}")

    for reg_type in ["l1", "l2", "elastic_net"]:
        # training with regularization
        theta_reg, beta_reg, val_acc_lst_reg, train_log_likelihoods_reg, val_log_likelihoods_reg = regular_irt(
            train_data, val_data, learning_rate, iterations, lambda_theta, lambda_beta, reg_type
        )

        # plot training and validation log-likelihood vs interation, compare the regularize with original
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)

        # plot training and validation log-likelihood vs interation
        plt.plot(range(1, iterations + 1), train_log_likelihoods, label="Training Log-Likelihood (Original)")
        plt.plot(range(1, iterations + 1), train_log_likelihoods_reg, label="Training Log-Likelihood (Regularized)")
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title("Training Log-Likelihood vs Iteration")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, iterations + 1), val_log_likelihoods, label="Validation Log-Likelihood (Original)")
        plt.plot(range(1, iterations + 1), val_log_likelihoods_reg, label="Validation Log-Likelihood (Regularized)")
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title("Validation Log-Likelihood vs Iteration")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # report final validation and test accuracies
        val_acc_reg = evaluate(val_data, theta_reg, beta_reg)
        test_acc_reg = evaluate(test_data, theta_reg, beta_reg)

        print(f"Final Validation Accuracy ({reg_type}): {val_acc_reg}")
        print(f"Final Test Accuracy ({reg_type}): {test_acc_reg}")


if __name__ == "__main__":
    main()
