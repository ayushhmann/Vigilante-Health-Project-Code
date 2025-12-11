# Train a sample RandomForest model using synthetic data resembling Vigilante Health features.
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Features idea: HR_mean, HR_std, BP_sys, SpO2, Temp, Sleep_hours, Activity_level, prakriti_V, prakriti_P, prakriti_K
def generate_synthetic(n_samples=1000, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=8, n_redundant=0, n_informative=6,
                               n_classes=3, n_clusters_per_class=1, random_state=random_state)
    cols = ['feat_'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    # create some interpretable columns by scaling
    df['HR_mean'] = 60 + (df['feat_0'] * 5)
    df['HR_std'] = np.abs(df['feat_1']) * 2
    df['BP_sys'] = 110 + (df['feat_2'] * 8)
    df['SpO2'] = 96 + (df['feat_3'] * 0.5)
    df['Temp'] = 36.5 + (df['feat_4'] * 0.1)
    df['Sleep_hours'] = 7 + (df['feat_5'] * 0.5)
    df['Activity_level'] = (df['feat_6'] - df['feat_7'])  # synthetic activity metric
    # prakriti one-hot simulation
    rng = np.random.RandomState(random_state)
    prak = rng.choice(['V','P','K'], size=n_samples)
    df['prakriti_V'] = (prak=='V').astype(int)
    df['prakriti_P'] = (prak=='P').astype(int)
    df['prakriti_K'] = (prak=='K').astype(int)
    # target y (0,1,2) from generated y
    df['risk'] = y
    # pick useful cols
    use_cols = ['HR_mean','HR_std','BP_sys','SpO2','Temp','Sleep_hours','Activity_level','prakriti_V','prakriti_P','prakriti_K','risk']
    return df[use_cols]

def train_and_save(df, model_path='rf_model.pkl'):
    X = df.drop(columns=['risk'])
    y = df['risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    # save model and feature order
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': X.columns.tolist()}, f)
    # save sample test CSV
    X_test.assign(risk=y_test).to_csv('sample_test_data.csv', index=False)
    print('Model and sample data saved.')

if __name__ == '__main__':
    df = generate_synthetic(800)
    df.to_csv('synthetic_vigilante_data.csv', index=False)
    train_and_save(df, model_path='rf_model.pkl')
