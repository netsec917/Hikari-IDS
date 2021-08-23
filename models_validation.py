import numpy as np  
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Innit tflite interpreter for model
def init_interpreter(model):
    interpreter = tf.lite.Interpreter(model)
    interpreter.allocate_tensors()
    # obtaining the input-output shapes and types
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def predict(X, interpreter, input_details, output_details):
    predictions = []
    for i in range(X.shape[0]):
        instance = X[i, :].reshape(input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], instance)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(prediction)

        if predictions[i] >= 0.50:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return np.array(predictions)


# Calculate metrics
def evaluate_predictions(y_val, predictions):
    accuracy = accuracy_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    return accuracy, recall, precision, f1


def run_model(model):
    interpreter, input_details, output_details = init_interpreter(model)
    start = time.process_time()
    predictions = predict(X_val, interpreter, input_details, output_details)
    cpu_time = time.process_time() - start
    accuracy, recall, precision, f1 = evaluate_predictions(y_val, predictions)
    return cpu_time, accuracy, recall, precision, f1, predictions


if __name__ == '__main__':
    X_val = np.load('./prepared/X_val.npy').astype('float32')
    y_val = np.load('./prepared/y_val.npy').astype('float32')
    models = []
    accuracies = []
    precisions = []
    recalls = []
    fscores = []
    deltas = []
    n_neurons = []
    metrics_df = pd.DataFrame(columns = ['Neurons', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'CPU Time'])


    cpu_time, accuracy, recall, precision, f1, predictions = run_model("models/sigmoid_2layer_25neurons.tflite")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
    n_neurons.append(25)
    confusion_mat_25 = confusion_matrix(y_val, predictions)


    cpu_time, accuracy, recall, precision, f1, predictions = run_model("models/sigmoid_2layer_50neurons.tflite")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
    n_neurons.append(50)
    confusion_mat_50 = confusion_matrix(y_val, predictions)


    cpu_time, accuracy, recall, precision, f1, predictions = run_model("models/sigmoid_2layer_75neurons.tflite")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
    n_neurons.append(75)
    confusion_mat_75 = confusion_matrix(y_val, predictions)


    cpu_time, accuracy, recall, precision, f1, predictions = run_model("models/sigmoid_2layer_125neurons.tflite")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
    n_neurons.append(125)
    confusion_mat_125 = confusion_matrix(y_val, predictions)


    cpu_time, accuracy, recall, precision, f1, predictions = run_model("models/sigmoid_2layer_150neurons.tflite")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
    n_neurons.append(150)
    confusion_mat_150 = confusion_matrix(y_val, predictions)


    cpu_time, accuracy, recall, precision, f1, predictions = run_model("models/sigmoid_2layer_175neurons.tflite")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
    n_neurons.append(175)
    confusion_mat_175 = confusion_matrix(y_val, predictions)

    #save results .csv
    metrics_df["Neurons"] = n_neurons
    metrics_df["Accuracy"] = accuracies
    metrics_df["Recall"] = recalls 
    metrics_df["Precision"] = precisions 
    metrics_df["F-Score"] = fscores 
    metrics_df["CPU Time"] = deltas
    metrics_df.to_csv('./results/sigmoid_2_layer_validation.csv')
    

  
    # Generate gridplot of confusion matrices
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    img = sns.heatmap(confusion_mat_25, annot = True, fmt = "d")
    img.set_title("25 Neurons")
    img.set_ylabel("True label")
    img.set_xlabel("Predicted label")
    
    ax = fig.add_subplot(2, 3, 2)
    img = sns.heatmap(confusion_mat_50, annot = True, fmt = "d")
    img.set_title("50 Neurons")
    img.set_ylabel("True label")
    img.set_xlabel("Predicted label")

    ax = fig.add_subplot(2, 3, 3)
    img = sns.heatmap(confusion_mat_75, annot = True, fmt = "d")
    img.set_title("75 Neurons")
    img.set_ylabel("True label")
    img.set_xlabel("Predicted label")

    ax = fig.add_subplot(2, 3, 4)
    img = sns.heatmap(confusion_mat_125, annot = True, fmt = "d")
    img.set_title("125 Neurons")
    img.set_ylabel("True label")
    img.set_xlabel("Predicted label")

    ax = fig.add_subplot(2, 3, 5)
    img = sns.heatmap(confusion_mat_150, annot = True, fmt = "d")
    img.set_title("150 Neurons")
    img.set_ylabel("True label")
    img.set_xlabel("Predicted label")

    ax = fig.add_subplot(2, 3, 6)
    img = sns.heatmap(confusion_mat_175, annot = True, fmt = "d")
    img.set_title("175 Neurons")
    img.set_ylabel("True label")
    img.set_xlabel("Predicted label")

    fig.suptitle("Confusion Matrices: 1 hidden layer (Sigmoid)")
    plt.tight_layout()
    plt.savefig('./results/confusion_matrices_1_hidden_sigmoid')