# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['bookId', 'title', 'author'],
    usecols=['bookId', 'title', 'author'],
    dtype={'bookId': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['userId', 'bookId', 'rating'],
    usecols=['userId', 'bookId', 'rating'],
    dtype={'userId': 'int32', 'bookId': 'str', 'rating': 'float32'})

# Clean data

# Get list of users to remove
user_ratingCount = (df_ratings.
     groupby(by = ['userId'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['userId', 'totalRatingCount']]
)
users_to_remove = user_ratingCount.query('totalRatingCount > 200').userId.tolist() # Remove users with less than 200 ratings

# merge rating and catalog by bookId
df = pd.merge(df_ratings,df_books,on='bookId')

# create totalRatingCount
book_ratingCount = (df.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )

rating_with_totalRatingCount = df.merge(book_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# remove books with less than 100 ratings
rating_popular_books = rating_with_totalRatingCount.query('totalRatingCount > 100')

# remove from the dataset users with less than 200 ratings
rating_popular_books = rating_popular_books[rating_popular_books['userId'].isin(users_to_remove)]

# pivot table and create matrix
book_features_df = rating_popular_books.pivot_table(index='title',columns='userId',values='rating').fillna(0)
book_features_df_matrix = csr_matrix(book_features_df.values)

def get_recommends(books):
    model_knn = NearestNeighbors(metric = 'cosine', n_neighbors=5, algorithm='auto')
    model_knn.fit(book_features_df_matrix)

    # Search for book by UserID
    for query_index in range(len(book_features_df)):
        if book_features_df.index[query_index] == books:
            break

    # creating return structure
    ret = [book_features_df.index[query_index], []]
    distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index,:].values.reshape(1, -1))
    # Return the recomendations
    for i in range(1, len(distances.flatten())):
        ret[1].insert(0, [book_features_df.index[indices.flatten()[i]], distances.flatten()[i]])
    return ret

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
