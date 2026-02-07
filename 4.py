"""
Machine Learning Model - Spam Email Detection
Author: [Your Name]
Date: February 2026
Project: Internship Assignment - ML Model Implementation

Description:
This script demonstrates a complete machine learning workflow using scikit-learn
to build a spam email classifier. Includes data preprocessing, model training,
evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(file_path):
    """
    Load the spam email dataset
    
    Parameters:
        file_path (str): Path to CSV file
        
    Returns:
        DataFrame: Loaded data
    """
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(file_path)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Total emails: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def explore_data(df):
    """
    Perform exploratory data analysis
    
    Parameters:
        df (DataFrame): Email dataset
    """
    print("\n" + "=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Display first few rows
    print("\nFirst 5 emails:")
    print(df.head())
    
    # Class distribution
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    
    # Percentage
    print("\nPercentage Distribution:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    # Basic statistics
    print("\nDataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    return df


def visualize_data(df):
    """
    Create visualizations of the data
    
    Parameters:
        df (DataFrame): Email dataset
    """
    print("\n" + "=" * 60)
    print("STEP 3: DATA VISUALIZATION")
    print("=" * 60)
    
    # Class distribution bar chart
    plt.figure(figsize=(10, 6))
    
    counts = df['label'].value_counts()
    plt.bar(counts.index, counts.values, color=['green', 'red'], alpha=0.7, edgecolor='black')
    plt.title('Distribution of Spam vs Ham Emails', fontsize=16, fontweight='bold')
    plt.xlabel('Email Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add value labels on bars
    for i, v in enumerate(counts.values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Class distribution chart saved: class_distribution.png")
    plt.close()
    
    # Message length analysis
    df['message_length'] = df['message'].apply(len)
    
    plt.figure(figsize=(12, 6))
    
    spam_lengths = df[df['label'] == 'spam']['message_length']
    ham_lengths = df[df['label'] == 'ham']['message_length']
    
    plt.hist(spam_lengths, bins=30, alpha=0.7, label='Spam', color='red', edgecolor='black')
    plt.hist(ham_lengths, bins=30, alpha=0.7, label='Ham', color='green', edgecolor='black')
    
    plt.title('Message Length Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Message Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('message_length_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Message length chart saved: message_length_distribution.png")
    plt.close()


def preprocess_data(df):
    """
    Preprocess the text data
    
    Parameters:
        df (DataFrame): Email dataset
        
    Returns:
        X, y: Features and labels
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATA PREPROCESSING")
    print("=" * 60)
    
    # Separate features and labels
    X = df['message']
    y = df['label']
    
    # Convert labels to binary (0: ham, 1: spam)
    y = y.map({'ham': 0, 'spam': 1})
    
    print("\n✓ Labels encoded:")
    print("  ham  -> 0")
    print("  spam -> 1")
    
    return X, y


def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    
    Parameters:
        X: Features
        y: Labels
        test_size: Proportion for testing
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("STEP 5: SPLITTING DATA")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set: {len(X_train)} emails ({(1-test_size)*100:.0f}%)")
    print(f"  Testing set:  {len(X_test)} emails ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Convert text to numerical features using TF-IDF
    
    Parameters:
        X_train: Training text
        X_test: Testing text
        
    Returns:
        X_train_vec, X_test_vec, vectorizer
    """
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE EXTRACTION (TF-IDF)")
    print("=" * 60)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=3000,  # Keep top 3000 features
        stop_words='english',  # Remove common English words
        lowercase=True,
        ngram_range=(1, 2)  # Use single words and pairs
    )
    
    # Transform text to TF-IDF features
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"\n✓ Text vectorization completed!")
    print(f"  Feature matrix shape: {X_train_vec.shape}")
    print(f"  Number of features: {len(vectorizer.get_feature_names_out())}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Show top features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n  Sample features: {list(feature_names[:10])}")
    
    return X_train_vec, X_test_vec, vectorizer


def train_models(X_train, y_train):
    """
    Train multiple machine learning models
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Trained models
    """
    print("\n" + "=" * 60)
    print("STEP 7: TRAINING MODELS")
    print("=" * 60)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained successfully!")
    
    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and compare performance
    
    Parameters:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame: Results comparison
    """
    print("\n" + "=" * 60)
    print("STEP 8: MODEL EVALUATION")
    print("=" * 60)
    
    results = []
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df


def plot_model_comparison(results_df):
    """
    Create visualization comparing model performance
    
    Parameters:
        results_df: DataFrame with model results
    """
    print("\n" + "=" * 60)
    print("STEP 9: VISUALIZING RESULTS")
    print("=" * 60)
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(results_df))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['skyblue', 'lightgreen', 'coral', 'gold']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, results_df[metric], width, 
                     label=metric, color=color, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison chart saved: model_comparison.png")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Create confusion matrix for the best model
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    """
    print("\n" + "=" * 60)
    print("STEP 10: CONFUSION MATRIX")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: confusion_matrix.png")
    plt.close()
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


def test_predictions(model, vectorizer, sample_messages):
    """
    Test the model with new messages
    
    Parameters:
        model: Trained model
        vectorizer: Fitted vectorizer
        sample_messages: List of messages to test
    """
    print("\n" + "=" * 60)
    print("STEP 11: TESTING WITH NEW MESSAGES")
    print("=" * 60)
    
    for msg in sample_messages:
        # Vectorize the message
        msg_vec = vectorizer.transform([msg])
        
        # Predict
        prediction = model.predict(msg_vec)[0]
        probability = model.predict_proba(msg_vec)[0]
        
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = probability[prediction] * 100
        
        print(f"\nMessage: \"{msg}\"")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")


def main():
    """Main execution function"""
    
    print("\n" + "=" * 60)
    print("   SPAM EMAIL DETECTION - MACHINE LEARNING MODEL")
    print("=" * 60)
    
    # Load data
    df = load_data('spam_emails.csv')
    
    # Explore data
    explore_data(df)
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    # Train models
    models = train_models(X_train_vec, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test_vec, y_test)
    
    # Visualize comparison
    plot_model_comparison(results)
    
    # Find best model
    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    best_model = models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {results.loc[results['Accuracy'].idxmax(), 'Accuracy']:.4f}")
    
    # Confusion matrix for best model
    plot_confusion_matrix(best_model, X_test_vec, y_test, best_model_name)
    
    # Test with new messages
    test_messages = [
        "Congratulations! You won $1 million! Click here now!",
        "Let's schedule a meeting for tomorrow at 2pm",
        "URGENT: Your account has been compromised! Act now!",
        "Thanks for your help with the project"
    ]
    
    test_predictions(best_model, vectorizer, test_messages)
    
    # Save the best model
    import pickle
    with open('spam_detector_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  📊 class_distribution.png")
    print("  📊 message_length_distribution.png")
    print("  📊 model_comparison.png")
    print("  📊 confusion_matrix.png")
    print("  💾 spam_detector_model.pkl")
    print("  💾 vectorizer.pkl")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
