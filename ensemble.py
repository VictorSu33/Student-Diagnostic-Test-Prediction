from utils import load_train_csv, load_valid_csv, load_public_test_csv
import numpy as np
from item_response import *

np.random.seed(1024)

def bootstrap(data):
    indices = np.random.randint(len(data["user_id"]), size=len(data["user_id"]))
    
    new_data = {"user_id": [0] *len(data["user_id"]) , "question_id": [0] *len(data["user_id"]), "is_correct": [0] *len(data["user_id"])}
    for i in indices:
        new_data["user_id"][i] = data["user_id"][i]
        new_data["question_id"][i] = data["question_id"][i]
        new_data["is_correct"][i] = data["is_correct"][i]

    return new_data
#choose model to train

def predict_irt(train_data, val_data, lr, iterations):
    theta, beta = irt(train_data,val_data,lr,iterations)[:2]

    pred = []
    for i, q in enumerate(val_data["question_id"]):
        u = val_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(int(p_a >= 0.5))
    return pred

def aggregate(train_data, val_data, lr, iterations):
    train_1 = bootstrap(train_data)
    train_2 = bootstrap(train_data)
    train_3 = bootstrap(train_data)

    #train models and evalute on val_data
    pred_1 = predict_irt(train_1,val_data,lr,iterations)
    pred_2 = predict_irt(train_2,val_data,lr,iterations)
    pred_3 = predict_irt(train_3,val_data,lr,iterations)

    #average the predictions of the three models
    avg_preds = []

    for i in range(len(pred_1)):
        avg = (pred_1[i] + pred_2[i] + pred_3[i]) / 3
        avg_preds.append(avg >= 0.5)

    accuracy = np.mean(np.array(val_data["is_correct"]) == np.array(avg_preds))
    return accuracy

def main():
    train_data = load_train_csv(r"ProjectDefault\data")
    val_data = load_valid_csv(r"ProjectDefault\data")
    test_data = load_public_test_csv(r"ProjectDefault\data")

    lr = 0.005
    iterations = 100

    val_acc = aggregate(train_data, val_data, lr, iterations)
    print(f"Validation Accuracy: {val_acc}")

    test_acc = aggregate(train_data, test_data, lr, iterations)
    print(f"Test Accuracy: {test_acc}")

if __name__ == "__main__":
    main()


