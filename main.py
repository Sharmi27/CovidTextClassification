import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Corona_NLP_test.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Check the column names
print("Column names:", df.columns)

# Assuming the column names are 'OriginalTweet' for the tweet text and 'Sentiment' for the tweet labels
tweet_column = 'OriginalTweet'
label_column = 'Sentiment'

# Split the dataset into features (X) and target (y)
X = df[tweet_column]
y = df[label_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')

# Train the classifier
svm_classifier.fit(tfidf_train, y_train)

# Predict on the test data
y_pred = svm_classifier.predict(tfidf_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
