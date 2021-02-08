# ------ Functions ------
# The following code is a list of functions written for the Breaking Hits capstone project.
# import packages
import mysql.connector
import pandas as pd
import numpy as np
import networkx as net
from surprise import SVD, Dataset, Reader
from scipy.sparse import csr_matrix
# -----------------------------------------------------
# ----- GENERAL DATA STUFF ----------------------------
# -----------------------------------------------------
def show_tables():
# This function connects to Breaking Hits MySQL server and
# prints out all tables in the database. Used as an initial resource.
# (May be removed later)
  cnx = mysql.connector.connect(user='breaking_read',
                                password='(hHy;gTPMnet',
                                host='206.225.82.147',
                                database='breaking_livedb')
  cur = cnx.cursor()
  cur.execute("SHOW TABLES")
  tables = cur.fetchall()
  cnx.close()
  # list all schemas available from the database
  [print(tables[i][0]) for i in range(len(tables))]
# -----------------------------------------------------
def pull_dataframe(query):
# This function connects to Breaking Hits MySQL server and converts the
# quieried table from the breaking_livedb schema.

# Here the inputs are as follows:
# query - the MySQL query you want to pull from the MySQL server

# EX: SELECT * FROM breaking_livedb.users;
# For a list of the MySQL tables use the show_tables() function.
  cnx = mysql.connector.connect(user='breaking_read',
                                password='(hHy;gTPMnet',
                                host='206.225.82.147',
                                database='breaking_livedb')
  cur = cnx.cursor()
  cur.execute(query)
  rows = cur.fetchall()                         # get all selected rows
  field_names = [i[0] for i in cur.description] # get all column names
  return(pd.DataFrame(rows, columns=field_names))
  cnx.close()
# -----------------------------------------------------
def users():
# This function connects to Breaking Hits MySQL server and pulls the users data
# and returns it as a pandas dataframe.
# The output is for TWO dataframes:
# the first one is for the listeners
# the second one for the dataframe of artists.
  cnx = mysql.connector.connect(user='breaking_read',
                                password='(hHy;gTPMnet',
                                host='206.225.82.147',
                                database='breaking_livedb')
  cur = cnx.cursor()
  cur.execute('SELECT * FROM breaking_livedb.users')
  rows = cur.fetchall()                         # get all selected rows
  field_names = [i[0] for i in cur.description] # get all column names
  cnx.close() # close the sql connection

  all_users = pd.DataFrame(rows, columns=field_names)
  listeners = all_users[all_users['is_artist'] == 0]
  artists = all_users[all_users['is_artist'] == 1]
  return(listeners, artists)
# -----------------------------------------------------
def get_ratings():
# This functions returns the user_music_votes (song ratings) data table from BH MySQL database
  rating = pull_dataframe('select * from user_music_votes').drop(['id','date_added'], axis =1)
  return(rating)
# -----------------------------------------------------
def get_spotlights():
# This functions returns the user_spotlights (songs spotlighted by a user) data table from BH MySQL database
  spotlights = pull_dataframe('select * from user_spotlight').drop(['id','date_added'], axis =1)
  return(spotlights)
# -----------------------------------------------------
def get_saves(date=False):
# This functions returns the user_saves (song saved by a user) data table from BH MySQL database
  saves = pull_dataframe('select * from user_saved').drop(['id','date_saved'], axis =1)
  user_id = saves['user_id']
  saves.drop(['user_id'], inplace=True, axis=1)
  saves = pd.concat([user_id, saves], axis=1)
  return(saves)
# -----------------------------------------------------------
# ----- SOCIAL NETWORK ANALYSIS------------------------------
# -----------------------------------------------------------    
#calculates the number and percentage of NaNs and empty cells by column in dataframe
def countMissing(df):
    count = (df.isnull().sum() + df.apply(lambda x: x.eq("").sum(), axis=0)) 
    perc = round(count/df.shape[0]*100,2)
    missing_df = pd.concat([count, perc], axis = 1)
    missing_df.columns =['Count', 'Percentage']
    return(missing_df)
# -----------------------------------------------------------   
#draws a network of followers and followed. Arguments passed into function 
#are data frames containing follower and followed columns. Returns a graph object.
def follow_net(df):
    g = net.Graph()
    gb = df.groupby("follower")
    groups =dict(list(gb))
    keys = groups.keys()
    for i in range(0,len(list(keys))): 
        for j in range(0,len(groups[list(keys)[i]]['follower'])):
            g.add_edge(list(keys)[i],groups[list(keys)[i]]['followed'].iloc[j])
            j = j + 1
        i = i + 1
    return (g)
# -----------------------------------------------------
# ----- RECOMMENDATION SYSTEM -------------------------
# -----------------------------------------------------
def bh_model(ratings):
# This function takes the proper list of ratings and computes the results...
# the parameters for the Single Value Decomposition function were tuned from
# gridsearch from the surprise package. (Collaborative Filtering)
#
# INPUT:
# ratings - dataframe of user, item, and ratings
#
# OUTPUT:
# rec - scikit 'surprise' package formatted list of tuples as a result.
#
# NOTE: the output of this function is designed to then to be inputted into the function rec() to
#       produce a proper prediction result for a specified user.
#
# establish the model and parameters
  svd = SVD(n_factors = 10, init_mean=0,
        init_std_dev=0.1, n_epochs=20,
        lr_all=0.003, reg_all=0.4,
        biased=True)

  # A reader is still needed but only the rating_scale param is requiered.
  reader = Reader(rating_scale=(1, 5))
  # The columns must correspond to user id, item id and ratings (in that order).
  data = Dataset.load_from_df(ratings[['user_id', 'user_music_id', 'rating']], reader)
  # create trainset for surprise formatting
  trainset = data.build_full_trainset()
  # create the test set from the data
  testset = trainset.build_testset()

  from surprise import accuracy
  svd.train(trainset) # train the model
  accuracy.rmse(svd.test(testset)) # predictions score - rmse
  return(svd) # returns 
# -----------------------------------------------------
def bh_predict(model, user_id, ratings):
# This function takes the results from the rec_model() function and returns a list
# of predictions unique for a particular user.
  rated_song_ids = ratings['user_music_id'].drop_duplicates().sort_values()
  predictions = []
  for i in rated_song_ids:
    predictions.append(model.predict(user_id,i))
  return(predictions)
# -----------------------------------------------------
def bh_top_recs(predictions):
# This function takes the results from the recommendation calculation and returns the recommendations
#
# INPUT:
# user_id - the specified user_id to make a recommendation to
# X - the list of tupled results (from the pred_rec() function)
# n - the number of recommendations to be returned
# rm_rating - If True -> function only returns the recommended song ID's
#       If False -> function returns the recommended song ID's and their  predicted ratings
#
# OUTPUT:
# returns a pandas dataframe of recommended songs
  rec_dict= {'Song_ID':[],
            'rating':[]}
  for i in range(len(predictions)):
    rec_dict['Song_ID'].append(predictions[i][1])
    rec_dict['rating'].append(predictions[i][3])
  res = pd.DataFrame(rec_dict)
  # Here we need to remove recommended songs for artists that are their won songs
  return(res)
# -----------------------------------------------------
def bh_song_names(recs):
# Takes the results from rec() function and returns the actual song name instead of Song_ID.
    musics = pull_dataframe('SELECT * FROM user_musics')[['id','title']]
    return(musics.merge(recs, how='right', left_on='id', right_on='Song_ID').drop(['id','Song_ID'], axis=1))
# -----------------------------------------------------
# ----Implicit Collaborative Filtering-----------------
# -----------------------------------------------------
def get_sparse(s):
    # function that creates the sparse matrix
    X = s
    s.replace('', np.nan,inplace=True)
    s.dropna(axis=0, inplace=True)
    X['rating'] = np.ones((len(X),1))
    
    user_list = pull_dataframe('select * from users')['id']
    song_list = pull_dataframe('select * from user_musics')['id']
    
    ratemat = pd.DataFrame(np.zeros((len(user_list), len(song_list))), index=list(user_list), columns=list(song_list))

    for i in range(len(s)):
        if (float(s.iloc[i,0]) in list(user_list)):
            if (float(s.iloc[i,1]) in list(song_list)):
                ratemat.loc[float(s.iloc[i,0]), float(s.iloc[i,1])] = 1;
    
    return(ratemat)
#-----------------------------------------------------
def jaccard_similarities(sparse_data_frame):
    mat = csr_matrix(sparse_data_frame)
    cols_sum = mat.getnnz(axis=0)
    ab = mat.T * mat

    # for rows
    aa = np.repeat(cols_sum, ab.getnnz(axis=0))
    # for columns
    bb = cols_sum[ab.indices]

    similarities = ab.copy()
    similarities.data = similarities.data / (aa + bb - ab.data)
    sim = pd.DataFrame(data=similarities.toarray(), index= sparse_data_frame.columns, columns= sparse_data_frame.columns)
    return(sim)
#-----------------------------------------------------
def jaccard_implicit_recommender(sparse_matrix,user):
# This function returns the similarities scores for a given user from
# a given spase matrix...

  # prelimnary calculations
  data_matrix2 = jaccard_similarities(sparse_matrix);
  data2 = sparse_matrix.copy()
  data2['user'] = sparse_matrix.index.values
  data_matrix2 = jaccard_similarities(sparse_matrix)
  data_items2 = sparse_matrix.copy()
  
  # The id of the user for whom we want to generate recommendations
  user_index = data2[data2.user == user].index.tolist()[0] # Get the frame index
  
  # Get the artists the user has likd
  known_user_likes = data_items2.loc[user_index,:]
  known_user_likes = known_user_likes[known_user_likes >0].index.values

  # Users likes for all items as a sparse vector.
  user_rating_vector = data_items2.loc[user_index,:]

  # Calculate the score.
  score = data_matrix2.dot(user_rating_vector).div(data_matrix2.sum(axis=1))

  # Remove the known likes from the recommendation.
  score = score.drop(known_user_likes)

  # Print the known likes and the top 20 recommendations.
  print(known_user_likes)
  return(score)
#-----------------------------------------------------
def ensemble_implicit_recommendations(a,b,user,drop=True):
# This function returns the averaged similarity score
# between spare matrix a and b.

  # initial calculations for a
  a_data_matrix2 = jaccard_similarities(a);
  a_data2 = a.copy()
  a_data2['user'] = a.index.values
  a_data_matrix2 = jaccard_similarities(a)
  a_data_items2 = a.copy()
  
  # initial calculations for b
  b_data_matrix2 = jaccard_similarities(b);
  b_data2 = b.copy()
  b_data2['user'] = b.index.values
  b_data_matrix2 = jaccard_similarities(b)
  b_data_items2 = b.copy()
  
  # get the scores
  a_user_index = a_data2[a_data2.user == user].index.tolist()[0]
  a_user_rating_vector = a_data_items2.loc[a_user_index,:]
  a_score = a_data_matrix2.dot(a_user_rating_vector).div(a_data_matrix2.sum(axis=1))
  
  b_user_index = b_data2[b_data2.user == user].index.tolist()[0]
  b_user_rating_vector = b_data_items2.loc[b_user_index,:]
  b_score = b_data_matrix2.dot(b_user_rating_vector).div(b_data_matrix2.sum(axis=1))
  
  # average score ensemble of the results of sparse a and sparse b
  ensemble = (a_score + b_score)
  
  # Get the artists the user has likd
  a_known_user_likes = a.loc[a_user_index,:]
  a_known_user_likes = a_known_user_likes[a_known_user_likes >0].index.values
  
  b_known_user_likes = b.loc[b_user_index,:]
  b_known_user_likes = b_known_user_likes[b_known_user_likes >0].index.values
  
  known_user_likes = np.unique(np.append(a_known_user_likes,b_known_user_likes))
  
  # drop the known spotlights and known saves
  
  # if drop == True then we drop spotlights and saves that the user has already made
  if drop == True:
      ensemble = ensemble.drop(known_user_likes)
  
  # here we replace the NA where in the jaccard calculation we divided by zero
  # (this makes no sense to leave empty as it has meaning being "implicit")
  ensemble.replace(np.nan, 0, inplace=True)
  return(pd.DataFrame(ensemble),known_user_likes)
#-----------------------------------------------------




