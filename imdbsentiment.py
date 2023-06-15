import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download the IMDb movie reviews dataset
nltk.download('movie_reviews')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the dataset
import random
random.shuffle(documents)

# Define a feature extractor function
def document_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in words)
    return features

# Get the most frequent words as features
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

# Extract features for each document
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the dataset into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test the classifier accuracy
print(f'Accuracy: {accuracy(classifier, test_set) * 100:.2f}%')

# Word Clouds for Positive and Negative Reviews
positive_reviews = ' '.join([' '.join(d) for d, category in documents if category == 'pos'])
negative_reviews = ' '.join([' '.join(d) for d, category in documents if category == 'neg'])

# Generate Word Cloud for Positive Reviews
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud - Positive Reviews')
plt.axis('off')
plt.show()

# Generate Word Cloud for Negative Reviews
wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud - Negative Reviews')
plt.axis('off')
plt.show()

# Confusion Matrix
from nltk.metrics import ConfusionMatrix
reference_set = [(category, classifier.classify(features)) for (features, category) in test_set]
test_set_labels = [c for (d, c) in test_set]
predicted_set = [classifier.classify(d) for (d, c) in test_set]
#cm = ConfusionMatrix(reference_set, predicted_set)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
cm = ConfusionMatrix(reference_set, predicted_set)
cm.plot()
plt.title('Confusion Matrix')
plt.show()

# # Confusion Matrix
# from nltk.metrics import ConfusionMatrix
# reference_set = [(c, 'pos') for (d, c) in test_set]
# test_set_labels = [c for (d, c) in test_set]
# predicted_set = [classifier.classify(d) for (d, c) in test_set]
# cm = ConfusionMatrix(reference_set, predicted_set)
#
# # Plot Confusion Matrix
# plt.figure(figsize=(6, 6))
# cm.plot()
# plt.title('Confusion Matrix')
# plt.show()