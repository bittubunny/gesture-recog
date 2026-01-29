import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

NUM_LANDMARKS = 21

def save_data(user_id, new_data):
    folder = "hand_sign_data"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{user_id}_data.csv")

    # Column names for landmarks
    landmark_cols = []
    for i in range(NUM_LANDMARKS):
        landmark_cols += [f"x{i}", f"y{i}", f"z{i}"]
    cols = landmark_cols + ["label"]

    # Create DataFrame with column names
    df_new = pd.DataFrame(new_data, columns=cols)

    if os.path.exists(path):
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)

    return path

def train_model(user_id):
    data_path = f"hand_sign_data/{user_id}_data.csv"
    if not os.path.exists(data_path):
        return None, "No data to train"

    df = pd.read_csv(data_path)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Save label map
    os.makedirs("labels", exist_ok=True)
    label_map_path = f"labels/{user_id}_labels.txt"
    with open(label_map_path, "w") as f:
        for label in encoder.classes_:
            f.write(label + "\n")

    # Simple train-test (all data for now)
    X_train, y_train = X, y_categorical

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_LANDMARKS*3,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{user_id}_model.h5"
    model.save(model_path)
    return model_path, "Training complete"

def load_labels(user_id):
    label_path = f"labels/{user_id}_labels.txt"
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    return []

def predict_hand_sign(user_id, landmarks):
    model_path = f"models/{user_id}_model.h5"
    if not os.path.exists(model_path):
        return None, "Model not trained"

    model = tf.keras.models.load_model(model_path)
    X_input = np.array(landmarks).reshape(1, -1)
    pred = model.predict(X_input, verbose=0)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100

    labels = load_labels(user_id)
    return f"{labels[class_idx]} ({confidence:.0f}%)", confidence
