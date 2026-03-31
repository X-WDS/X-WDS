import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Gymnasium + stable-baselines3
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ======================
# 1. Load Dataset (ONLY ONE DATASET)
# ======================
print("Loading Batalas dataset...")

def load_dataset():
    path = "data/BATADAL_dataset04.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")

    if "DATETIME" in df.columns:
        df.drop(columns=["DATETIME"], inplace=True)

    X = df.drop(columns=["ATT_FLAG"]).values
    y = df["ATT_FLAG"].values

    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y


try:
    X, y = load_dataset()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


# ======================
# 2. Train / Validation Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ======================
# 3. Train Initial Model
# ======================
print("\n" + "="*50)
print("TRAINING INITIAL XGBOOST MODEL")
print("="*50)

initial_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)

initial_model.fit(X_train_scaled, y_train)
y_pred_initial = initial_model.predict(X_test_scaled)
initial_accuracy = accuracy_score(y_test, y_pred_initial)

print(f"Initial Model Accuracy: {initial_accuracy:.4f}")
print(classification_report(y_test, y_pred_initial))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_initial),
            annot=True, fmt='d', cmap="Blues")
plt.title("Initial Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")

initial_model.save_model("in_xgb.json")


# ======================
# 4. Custom RL Environment
# ======================
class ModelImprovementEnv(gym.Env):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        self.action_space = gym.spaces.Box(
            low=np.array([0.01, 3, 50, 0.5], dtype=np.float32),
            high=np.array([0.3, 10, 200, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.initial_accuracy = initial_accuracy
        self.current_accuracy = initial_accuracy
        self.current_params = [0.1, 6, 100, 0.8]
        self.best_accuracy = initial_accuracy
        self.best_params = self.current_params.copy()
        self.step_count = 0
        self.max_steps = 15

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_accuracy = self.initial_accuracy
        self.current_params = [0.1, 6, 100, 0.8]
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        new_params = np.clip(action, self.action_space.low, self.action_space.high)

        try:
            model = xgb.XGBClassifier(
                learning_rate=float(new_params[0]),
                max_depth=int(new_params[1]),
                n_estimators=int(new_params[2]),
                subsample=float(new_params[3]),
                random_state=42,
                eval_metric="logloss",
                verbosity=0
            )

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            new_acc = accuracy_score(self.y_test, y_pred)

            reward = new_acc - self.current_accuracy
            self.current_accuracy = new_acc
            self.current_params = new_params.tolist()

            if new_acc > self.best_accuracy:
                self.best_accuracy = new_acc
                self.best_params = new_params.tolist()

            terminated = False
            truncated = self.step_count >= self.max_steps

        except Exception:
            reward = -1.0
            terminated = False
            truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array(
            [self.current_accuracy] + self.current_params,
            dtype=np.float32
        )


# ======================
# 5. Train RL Agent
# ======================
print("\n" + "="*50)
print("TRAINING RL AGENT FOR HYPERPARAMETER OPTIMIZATION")
print("="*50)

env = DummyVecEnv([
    lambda: ModelImprovementEnv(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )
])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=512
)

model.learn(total_timesteps=5000)

best_params = env.envs[0].best_params
best_acc = env.envs[0].best_accuracy

print(f"\nBest Params: {best_params}")
print(f"Best Accuracy: {best_acc:.4f}")


# ======================
# 6. Retrain with RL-Optimized Params
# ======================
improved_model = xgb.XGBClassifier(
    learning_rate=best_params[0],
    max_depth=int(best_params[1]),
    n_estimators=int(best_params[2]),
    subsample=best_params[3],
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)

improved_model.fit(X_train_scaled, y_train)
y_pred_improved = improved_model.predict(X_test_scaled)
improved_acc = accuracy_score(y_test, y_pred_improved)

print(f"\nImproved Model Accuracy: {improved_acc:.4f}")
print(classification_report(y_test, y_pred_improved))

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_improved),
            annot=True, fmt='d', cmap="Greens")
plt.title("Improved Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()


# ======================
# 7. SHAP Analysis
# ======================
print("\n" + "="*50)
print("EXPLAINABLE AI: SHAP ANALYSIS")
print("="*50)

try:
    explainer = shap.TreeExplainer(improved_model)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"SHAP skipped: {e}")


joblib.dump(scaler, "scaler.pkl")


# ======================
# 8. Save Results
# ======================
results_df = pd.DataFrame({
    "Model": ["Initial", "Improved"],
    "Accuracy": [initial_accuracy, improved_acc],
    "Learning_Rate": [0.1, best_params[0]],
    "Max_Depth": [6, int(best_params[1])],
    "N_Estimators": [100, int(best_params[2])],
    "Subsample": [1.0, best_params[3]]
})

results_df.to_csv("model_results.csv", index=False)
improved_model.save_model("improved_xgb.json")

print("\n✅ Done! Results saved to 'model_results.csv' and 'improved_xgb.json'")
