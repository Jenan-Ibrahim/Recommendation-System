import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time


# Load the dataset
dataset = pd.read_csv('/content/drive/MyDrive/dataset1.csv')

categories = ['graphic', 'Craft', 'politics', 'political', 'Mathematics', 'zoology', 'business', 'dance', 'HR management', 'art', 'science', 'banking', 'drawing', 'GST', 'programming', 'operating system', 'photography']

for category in categories:
    print("******************")
    print(f"Category: {category}")
    print("******************")

    # Initialize the SVM model
    svm_model = SVC(kernel='linear')

    # Filter the dataset for the current category
    category_data = dataset[dataset['category'] == category]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(category_data[['post_id', 'category', 'title']], category_data['post_id'], test_size=0.2, random_state=42)

    # Initialize a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Vectorize the post titles in the training data
    train_title_vectors = tfidf_vectorizer.fit_transform(X_train['title'])

    start_time = time.time()
    # Train the SVM model for the current category
    svm_model.fit(train_title_vectors, X_train['post_id'])
    execution_time = time.time() - start_time
    start_time1 = time.time()
    for user_id, title in zip(X_test['post_id'], X_test['title']):
        print(f"User id: {user_id}")
        print(f"Current post: {title}")
        print("Suggestions:")

        # Vectorize the title of the current post
        test_title_vector = tfidf_vectorizer.transform([title])

        # Predict the closest post using the SVM model
        predicted_post_id = svm_model.predict(test_title_vector)[0]

        # Find the post with the predicted_post_id in the category_data
        suggested_post = category_data[category_data['post_id'] == predicted_post_id]

        if not suggested_post.empty:
            for idx, row in suggested_post.iterrows():
                print(f"{idx + 1}. {row['title']}")

        print("------------------------------")
execution_time1 = time.time() - start_time1
print(f"Total testing time: {execution_time1} seconds") print(f"Total training  time: {execution_time} seconds")