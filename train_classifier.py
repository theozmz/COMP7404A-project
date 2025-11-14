import os
import argparse
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_classifier(human_features_path, gpt_features_path, save_path, seed=114514):
    """
    Train classifier from pre-computed features.
    
    Parameters:
        human_features_path: Path to human features pickle file
        gpt_features_path: Path to GPT features pickle file
        save_path: Path to save trained model
        seed: Random seed for reproducibility
    
    Returns:
        Trained classifier
    """
    print("Loading features...")
    with open(human_features_path, 'rb') as f:
        human_features = np.array(pickle.load(f))
    with open(gpt_features_path, 'rb') as f:
        gpt_features = np.array(pickle.load(f))
    
    # Combine features and labels
    # 0 = human, 1 = AI
    features = np.concatenate([human_features, gpt_features], axis=0)
    labels = np.concatenate([np.zeros(len(human_features)), np.ones(len(gpt_features))], axis=0)
    
    # Train classifier
    print("Training classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
    classifier.fit(features, labels)
    print("Training completed!")
    
    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {save_path}")
    
    return classifier


def load_classifier(model_path):
    """
    Load a trained classifier from file.
    
    Parameters:
        model_path: Path to saved classifier (.pkl file or directory containing classifier_model.pkl)
    
    Returns:
        Loaded classifier
    """
    # If model_path is a directory, try to find classifier_model.pkl
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, 'classifier_model.pkl')
        if os.path.exists(candidate):
            model_path = candidate
        else:
            raise FileNotFoundError(f"No classifier_model.pkl found in {model_path}")
    
    print(f"Loading classifier from {model_path}...")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    print("Classifier loaded successfully!")
    
    return classifier


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier from pre-computed features"
    )
    parser.add_argument('--train_features_dir', type=str, required=True,
                       help="Directory containing human_features.pkl and GPT_features.pkl")
    parser.add_argument('--train_task', type=str, required=True,
                       help="Task name for feature files (e.g., 'Arxiv', 'Code', 'Essay')")
    parser.add_argument('--output_path', type=str, required=True,
                       help="Path to save trained model (.pkl)")
    parser.add_argument('--seed', type=int, default=114514,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Construct feature file paths
    human_path = os.path.join(args.train_features_dir, f"{args.train_task}_human_features.pkl")
    gpt_path = os.path.join(args.train_features_dir, f"{args.train_task}_GPT_features.pkl")
    
    # Check if files exist
    if not os.path.exists(human_path):
        print(f"Error: Human features file not found: {human_path}")
        return
    
    if not os.path.exists(gpt_path):
        print(f"Error: GPT features file not found: {gpt_path}")
        return
    
    # Train and save classifier
    train_classifier(human_path, gpt_path, args.output_path, args.seed)
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

