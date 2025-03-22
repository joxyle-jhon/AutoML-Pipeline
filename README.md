# AutoML Breast Cancer Classification

This project demonstrates how to use **Auto-sklearn** for automated machine learning (AutoML) to classify breast cancer data. The script applies **AutoSklearnClassifier** to predict whether a tumor is malignant or benign using the **Breast Cancer** dataset from `sklearn`.

## 🚀 Features
- Load and preprocess the Breast Cancer dataset
- Perform train-test data splitting
- Build an AutoML classification model using Auto-sklearn
- Evaluate model accuracy and generate a classification report
- Display the best models using the Auto-sklearn leaderboard

## 🛠️ Requirements
Ensure you have Python 3.8 or later installed. The necessary dependencies are:

- `numpy`
- `pandas`
- `scikit-learn`
- `auto-sklearn`
- `joblib`

## 📦 Installation
Follow these steps to set up the environment:

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate # On Windows: env\Scripts\activate
    ```

2. **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn joblib auto-sklearn
    ```

## 🖋️ Usage
1. Clone or download this repository.
2. Navigate to the project directory.
3. Run the script using:
    ```bash
    python main.py
    ```

## 🧪 Example Output
After running, the program will display the following:
- Model accuracy
- Classification report
- Best models identified by Auto-sklearn

Example:
```
Accuracy: 0.97

Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       100
           1       0.97      0.96      0.96        80

Best Models:
      rank  ensemble_weight               type
         1             0.50    RandomForest
         2             0.30         XGBoost
```

## 🛡️ Troubleshooting
- **Memory Issues:** Auto-sklearn requires sufficient memory. Consider increasing RAM or using a cloud environment.
- **CMake or SWIG Errors:** Ensure they are installed using:
  ```bash
  pip install cmake swig
  ```
- **ModuleNotFoundError:** Confirm that the virtual environment is activated and dependencies are installed.

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Feel free to submit pull requests or raise issues for improvements!

---

Happy coding! 😊

