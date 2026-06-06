import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_heart_data

def train_and_save_model():
    print("Fetching data...")
    df = load_heart_data()
    
    # Split features (User Input) and Target (Prediction)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    # Stratified split to preserve class imbalance ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest Classifier...")
    # Using Random Forest (Excellent for structured tabular medical data)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Training Complete. Test Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # Export the trained model to a file
    model_filename = 'heart_failure_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model successfully saved as '{model_filename}'")
    return model

if __name__ == "__main__":
    train_and_save_model()