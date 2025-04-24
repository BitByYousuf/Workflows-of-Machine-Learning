[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/apFgjlIP)


# DNA Splice Junction Classifier

This project trains and evaluates a machine learning model to classify DNA splice junctions using a Gradient Boosting Classifier.

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python train.py
```

This trains the model and saves:
- `model.pkl` (trained model)
- `test_data.pkl` (test set)

---

## Run Evaluation

```bash
python test.py
```

This loads the model and test data and prints:
- Accuracy  
- Precision  
- Recall  
- F1 Score
