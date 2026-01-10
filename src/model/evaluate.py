from sklearn.metrics import f1_score, roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(classification_report(y_test, y_pred))
    return f1, auc

# Usage: f1, auc = evaluate_model(model, X_test, y_test)