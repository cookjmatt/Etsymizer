from flask import request
from flask import render_template
from flaskexample import app
from .model import *

@app.route('/')
def path_input():
	return render_template("input.html")

@app.route('/output')
def path_output():
	listing_id = request.args.get('listing_id')
	result = return_model_info(listing_id)
	return render_template("output.html", feature_list = result.get('feature_list'),
										  recommended_tags = result.get('recommended_feature_list'),
										  actual_price = "{0:.2f}".format(float(result.get('original_price'))),
										  predicted_price = "{0:.2f}".format(result.get('original_price_prediction')),
										  actual_views = "{0:.2f}".format(result.get('original_views')),
										  predicted_views = "{0:.2f}".format(result.get('original_views_prediction')),
										  similar_features_price_prediction = "{0:.2f}".format(result.get('recommended_price_prediction')),
										  similar_features_views_prediction = "{0:.2f}".format(result.get('recommended_views_prediction')),
										  original_id = result.get('original_id'))

@app.route('/output2')
def path_output2():
	new_features = request.args.get('new_features')
	listing_id = request.args.get('listing_id')
	result = return_model_info(listing_id)
	result2 = return_new_features(new_features)
	return render_template("output2.html", feature_list = result2.get('feature_list'),
										  recommended_tags = result2.get('recommended_feature_list'),
										  actual_price = "{0:.2f}".format(float(result2.get('original_price'))),
										  predicted_price = "{0:.2f}".format(result2.get('original_price_prediction')),
										  actual_views = "{0:.2f}".format(result2.get('original_views')),
										  predicted_views = "{0:.2f}".format(result.get('original_views_prediction')),
										  similar_features_price_prediction = "{0:.2f}".format(result2.get('recommended_price_prediction')),
										  similar_features_views_prediction = "{0:.2f}".format(result2.get('recommended_views_prediction')),
										  original_id = result.get('original_id'),
										  original_feature_list = result.get('feature_list'),
										  original_price = "{0:.2f}".format(float(result.get('original_price'))),
										  original_views = "{0:.2f}".format(result.get('original_views')),
										  original_predicted_price = "{0:.2f}".format(result.get('original_price_prediction')),
										  original_predicted_views = "{0:.2f}".format(result.get('original_views_prediction')),
										  new_features = result2.get('new_features'))