
# Defines and globals
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime

API_KEY = 'c1lq1mzoyp1z0nx6bsqi9lrk'
request_data = {'api_key':'c1lq1mzoyp1z0nx6bsqi9lrk'}

# Load models and data structures
views_est = joblib.load('tfidf_views_GradientBoostingRegressor.pkl') 
price_est = joblib.load('tfidf_price_GradientBoostingRegressor.pkl')
num_favorers_est = joblib.load('tfidf_num_favorers_GradientBoostingRegressor.pkl')
lda = joblib.load('tf_lda.pkl') 
nmf = joblib.load('tfidf_nmf.pkl')
listings = joblib.load('listings.pkl')
features = joblib.load('all_features.pickle')
tf_features = joblib.load('tf_features.pkl')
tfidf_features = joblib.load('tfidf_features.pkl')
tf_vectorizer = joblib.load('tf_vectorizer.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

tf_feature_names = tf_vectorizer.get_feature_names()
prices = np.nan_to_num(listings['price'].tolist())
views = np.nan_to_num(listings['views'].tolist())
favorers = np.nan_to_num(listings['num_favorers'].tolist())
test_size = 0.1

tfidf_price_train, tfidf_price_test, y_price_train, y_price_test = train_test_split(
    tfidf_features, prices, test_size = test_size)

tfidf_views_train, tfidf_views_test, y_views_train, y_views_test = train_test_split(
    tfidf_features, views, test_size = test_size)

n_features = 1000 # Number of features
n_samples = 36570 # Number of samples
n_components = 100 # Number of components
n_top_words = 10  # Number of top words

# Print the top words for each topic in a model (NMF or LDA in this case)
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Return a list of top features for a given topic group in a model
def model_topic_features(model, feature_names, n_top_words, topic_group):
    for topic_idx, topic in enumerate(model.components_):
        if (topic_idx == topic_group):
            result = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return(result)

def find_similar_features(model, features):
    topic_list = model.transform(tf_vectorizer.transform([features]))[0].argmax()
    topic_features = model_topic_features(model, tf_feature_names, 50, topic_list)
    new_features = [item for item in topic_features.split() if item not in features.split()]
    return(new_features)

# Get features (tags + materials) for a given Etsy ID
def get_features_from_listing_id (listing_id):
    request = str('https://openapi.etsy.com/v2/listings/'+str(listing_id))
    r = requests.get(request, request_data).json()
    tags = " ".join(r.get('results')[0].get('tags')).lower()
    materials = " ".join(r.get('results')[0].get('materials')).lower()
    features = tags+" "+materials
    return features

def find_similar_features_from_etsy_id(etsy_id, model):
    original_features = get_features_from_listing_id(etsy_id)
    features = " ".join(tfidf_vectorizer.inverse_transform(tfidf_vectorizer.transform([original_features]))[0])
    similar_features = " ".join(find_similar_features(model, features)[:10])
    return [features, similar_features]

# Get price, views, and number of favorers for a given Etsy ID
def get_listing_info(listing_id):
    request = str('https://openapi.etsy.com/v2/listings/'+str(listing_id))
    r = requests.get(request, request_data).json()
    original_creation_date = r.get('results')[0].get('original_creation_tsz')
    original_creation_date = datetime.strptime(datetime.utcfromtimestamp(original_creation_date).strftime('%Y-%m-%d %H:%M:%S'), "%Y-%m-%d %H:%M:%S")
    ending_date = r.get('results')[0].get('ending_tsz')
    ending_date = datetime.strptime(datetime.utcfromtimestamp(ending_date).strftime('%Y-%m-%d %H:%M:%S'), "%Y-%m-%d %H:%M:%S")
    difference = (ending_date - original_creation_date).days
    price = r.get('results')[0].get('price')
    views = r.get('results')[0].get('views')
    views = views / difference
    num_favorers = r.get('results')[0].get('num_favorers')
    num_favorers = num_favorers / difference
    return([price, views, num_favorers, difference])

def make_draggable_list(in_string):
    in_string = in_string.split()
    output = " "
    output = "<ol data-draggable=\"target\">"
    for entry in in_string:
        output += "<li data-draggable=\"item\">"+entry+"</li>"
    output += "</ol>"
    return output

def return_new_features(new_features):
    similar_features = " ".join(find_similar_features(lda, new_features))
    original_price_prediction = price_est.predict(tfidf_vectorizer.transform([new_features]))[0]
    original_price = original_price_prediction
    original_views_prediction = views_est.predict(tfidf_vectorizer.transform([new_features]))[0]
    original_views = original_views_prediction
    similar_features_price_prediction = price_est.predict(tfidf_vectorizer.transform([similar_features]))[0]
    similar_features_views_prediction = views_est.predict(tfidf_vectorizer.transform([similar_features]))[0]

    return {'feature_list': make_draggable_list(new_features),
            'original_price': original_price,
            'original_price_prediction': original_price_prediction,
            'original_views': original_views,
            'original_views_prediction': original_views_prediction,
            'recommended_feature_list': make_draggable_list(similar_features),
            'recommended_price_prediction': similar_features_price_prediction,
            'recommended_views_prediction': similar_features_views_prediction,
            'new_features': new_features}

def return_model_info(etsy_id):
    feature_info = find_similar_features_from_etsy_id(etsy_id, lda)
    features = feature_info[0]
    similar_features = feature_info[1]
    listing_info = get_listing_info(etsy_id)
    original_price = listing_info[0]
    original_price_prediction = price_est.predict(tfidf_vectorizer.transform([features]))[0]
    original_views = listing_info[1]
    original_views_prediction = views_est.predict(tfidf_vectorizer.transform([features]))[0]
    similar_features_price_prediction = price_est.predict(tfidf_vectorizer.transform([similar_features]))[0]
    similar_features_views_prediction = views_est.predict(tfidf_vectorizer.transform([similar_features]))[0]

    original_feature_list = make_draggable_list(feature_info[0])

    return {'feature_list': make_draggable_list(features),
            'original_price': original_price,
            'original_price_prediction': original_price_prediction,
            'original_views': original_views,
            'original_views_prediction': original_views_prediction,
            'recommended_feature_list': make_draggable_list(similar_features),
            'recommended_price_prediction': similar_features_price_prediction,
            'recommended_views_prediction': similar_features_views_prediction,
            'original_id': etsy_id}