import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM  # it will later create a model for us

# ##we gonna fetch our datasets

data = fetch_movielens(min_rating=4.0)

# the above method will create a interaction metrics from csv file and store it in data variable as a dictionary

# now printing training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp')   # warp = weighted Approximate-Rank Pairwise
# it gradient descent algorithm to iteratively find the weights and improve our predictions
# it uses user's past rating history
# Content + collaborative(similar user's ratings) = Hybrid
# train model
model.fit(data['train'],epochs=30,num_threads=2)
# the dataset we want to train it on + the no. of epochs we want to train it for + and the number
# of threads we want to run it on parameter in fit method respectively


# generate recommendation from our model
def sample_recommendation(model, data, user_ids):

      n_users , n_items = data['train'].shape

# number of users and movies in training data using the shape attribute in our dictionary we created

# generate recommendations for each user we input
      for user_id in user_ids:

            # we will generate recommendations and store them in the scores variable using the
            # predict of our model we will use the userID as the first parameter and then a list
            # of each movie then a range method of numpy will give us every number from 0 up to the
            # no of items so we can predict the score for every movie then we will sort them
            # in order of their score the arg sort method of numpy will return the score indices in
            # descending order thanks to the negative sign let's go ahead and print them

            #    movies they already like
           known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

           scores = model.predict(user_id, np.arange(n_items))

           top_items = data['item_labels'][np.argsort(-scores)]

           print("User %s"%user_id)
           print(" Known positives:")
           for x in known_positives[:3]:
                print("         %s" % x)
           print("       Recommended:")

           for x in top_items[:3]:
                print("        %s" % x)

sample_recommendation(model,data,[3, 25, 450])









