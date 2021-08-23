import numpy as np
from sklearn import metrics  
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def evaluate_predictions(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return accuracy, recall, precision, f1


def run_model(model):
    interpreter, input_details, output_details = init_interpreter(model)
    start = time.process_time()
    predictions = predict(X_test, interpreter, input_details, output_details)
    cpu_time = time.process_time() - start
    accuracy, recall, precision, f1 = evaluate_predictions(y_test, predictions)
    return cpu_time, accuracy, recall, precision, f1


if __name__ == '__main__':
    X_test = np.load('./prepared/X_test.npy').astype('float32')
    y_test = np.load('./prepared/y_test.npy').astype('float32')
    models = []
    accuracies = []
    precisions = []
    recalls = []
    fscores = []
    deltas = []
    metrics_df = pd.DataFrame(columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'CPU Time'])
    

    cpu_time, accuracy, recall, precision, f1 = run_model("models/baseline.tflite")
    models.append("Baseline")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)


    cpu_time, accuracy, recall, precision, f1 = run_model("models/quantized.tflite")
    models.append("Quantized")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)


    cpu_time, accuracy, recall, precision, f1 = run_model("models/clustered.tflite")
    models.append("Clustered")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)


    cpu_time, accuracy, recall, precision, f1 = run_model("models/pruned.tflite")
    models.append("Pruned")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)


    cpu_time, accuracy, recall, precision, f1 = run_model("models/clustered_quantized.tflite")
    models.append("Clustered and quantized")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)

    
    cpu_time, accuracy, recall, precision, f1 = run_model("models/pruned_quantized.tflite")
    models.append("Pruned and quantized")
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)
    fscores.append(f1)
    deltas.append(cpu_time)
 

    # save results .csv to disk
    metrics_df["Model"] = models
    metrics_df["Accuracy"] = accuracies
    metrics_df["Recall"] = recalls 
    metrics_df["Precision"] = precisions 
    metrics_df["F-Score"] = fscores 
    metrics_df["CPU Time"] = deltas
    metrics_df.to_csv('./results/test_metrics.csv')


    img = sns.barplot(x = models, y = fscores)
    plt.xticks(rotation = 45)
    img.set_title("Ton-IoT test set f-score")
    plt.tight_layout()
    plt.savefig('./results/test_set_fscore')


    img = sns.barplot(x = models, y = accuracies)
    plt.xticks(rotation = 45)
    img.set_title("Ton-IoT Test set accuracy")
    plt.tight_layout()
    plt.savefig('./results/test_set_accuracy')


    img = sns.barplot(x = models, y = precisions)
    plt.xticks(rotation = 45)
    img.set_title("Ton-IoT test set precision")
    plt.tight_layout()
    plt.savefig('./results/test_set_precision')


    img = sns.barplot(x = models, y = recalls)
    plt.xticks(rotation=45)
    img.set_title("Ton-IoT test set recall")
    plt.tight_layout()
    plt.savefig('./results/test_set_recall')