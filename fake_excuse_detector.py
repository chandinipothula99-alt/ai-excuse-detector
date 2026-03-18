# ============================================================
#   AI Fake Excuse Detector
#   A beginner-friendly ML project using scikit-learn
# ============================================================

# ── STEP 0: Install required libraries (run once in terminal)
# pip install scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# STEP 1: Load the Dataset
# ============================================================

# If you have the CSV file:
df = pd.read_csv("excuse_dataset.csv")

# --- OR paste the dataset directly in code ---
# data = {
#     "text": ["I was sick with fever", "My dog ate my homework", ...],
#     "label": ["genuine", "fake", ...]
# }
# df = pd.DataFrame(data)

print("=" * 50)
print("STEP 1: Dataset Loaded")
print("=" * 50)
print(f"Total examples : {len(df)}")
print(f"Genuine excuses: {(df['label'] == 'genuine').sum()}")
print(f"Fake excuses   : {(df['label'] == 'fake').sum()}")
print("\nSample rows:")
print(df.sample(5).to_string(index=False))
print()


# ============================================================
# STEP 2: Prepare the Data
# ============================================================

# X = input text,  y = label (genuine / fake)
X = df["text"]
y = df["label"]

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 50)
print("STEP 2: Data Split")
print("=" * 50)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")
print()


# ============================================================
# STEP 3: Convert Text to Numbers (TF-IDF)
# ============================================================
# Machines can't read text — we turn each sentence into a
# vector of numbers using TF-IDF (Term Frequency-Inverse
# Document Frequency).

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)   # learn + transform
X_test_vec  = vectorizer.transform(X_test)         # only transform

print("=" * 50)
print("STEP 3: Text Vectorized (TF-IDF)")
print("=" * 50)
print(f"Each sentence is now a vector of {X_train_vec.shape[1]} features")
print()


# ============================================================
# STEP 4: Train the Model
# ============================================================
# We'll train BOTH models and compare — pick whichever you like!

# --- Model A: Logistic Regression ---
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

# --- Model B: Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

print("=" * 50)
print("STEP 4: Models Trained ✓")
print("=" * 50)
print("  → Logistic Regression")
print("  → Naive Bayes (Multinomial)")
print()


# ============================================================
# STEP 5: Evaluate the Models
# ============================================================

lr_preds = lr_model.predict(X_test_vec)
nb_preds = nb_model.predict(X_test_vec)

print("=" * 50)
print("STEP 5: Model Evaluation")
print("=" * 50)

print(f"\n📊 Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds)*100:.1f}%")
print(classification_report(y_test, lr_preds))

print(f"📊 Naive Bayes Accuracy        : {accuracy_score(y_test, nb_preds)*100:.1f}%")
print(classification_report(y_test, nb_preds))


# ============================================================
# STEP 6: Test with Your Own Excuses!
# ============================================================

def predict_excuse(excuse_text, model=lr_model):
    """
    Takes a sentence and returns 'genuine' or 'fake'
    along with the confidence score.
    """
    # Step 1: Vectorize the input
    vec = vectorizer.transform([excuse_text])

    # Step 2: Predict the label
    prediction = model.predict(vec)[0]

    # Step 3: Get confidence (probability)
    proba = model.predict_proba(vec)[0]
    classes = model.classes_

    # Create dictionary like {'fake': 0.3, 'genuine': 0.7}
    prob_dict = {classes[i]: proba[i] * 100 for i in range(len(classes))}

    confidence = max(proba) * 100

    return prediction, confidence, prob_dict


print("=" * 50)
print("STEP 6: Testing with New Excuses")
print("=" * 50)

test_excuses = [
    "I had a severe headache and couldn't open my eyes",   # likely genuine
    "A dragon kidnapped my notebook",                       # likely fake
    "The traffic was really bad due to a roadblock",        # likely genuine
    "I forgot to do it because I was busy doing nothing",   # likely fake
    "My internet stopped working during the submission",    # likely genuine
]

for excuse in test_excuses:
    label, confidence, prob_dict = predict_excuse(excuse)
    emoji = "✅" if label == "genuine" else "🚨"
    print(f"\n  Excuse    : \"{excuse}\"")
    print(f"  Verdict   : {emoji} {label.upper()}")
    print(f"  Confidence: {confidence:.1f}%")


# ============================================================
# STEP 7: Interactive Mode — Try Your Own!
# ============================================================

print("\n" + "=" * 50)
print("STEP 7: Interactive Excuse Checker")
print("=" * 50)
print("Type an excuse and press Enter. Type 'quit' to stop.\n")

while True:
    user_input = input("Your excuse: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye! No more excuses. 😄")
        break

    if not user_input:
        continue

    label, confidence, prob_dict = predict_excuse(user_input)
    emoji = "✅" if label == "genuine" else "🚨"

    

    # -------- Explanation (NEW FEATURE) --------
    input_vector = vectorizer.transform([user_input])
    feature_names = vectorizer.get_feature_names_out()
    input_array = input_vector.toarray()[0]

    top_indices = input_array.argsort()[-3:]
    important_words = [feature_names[i] for i in top_indices if input_array[i] > 0]
    print("\n" + "="*40)
    print("RESULT")
    print("="*40)

    print(f"Excuse: {user_input}\n")

    print(f"Verdict   : {emoji} {label.upper()}")
    print(f"Confidence: {confidence:.1f}%\n")

    print(f"Genuine Probability : {prob_dict.get('genuine', 0):.1f}%")
    print(f"Fake Probability    : {prob_dict.get('fake', 0):.1f}%")

    if confidence < 60:
        print("\n⚠️ Low confidence prediction — result may not be accurate")

    print(f"\nImportant words: {important_words}")
    print("="*40 + "\n")
   