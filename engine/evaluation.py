from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    return {
        "predictions": preds,
        "metrics": report
    }
