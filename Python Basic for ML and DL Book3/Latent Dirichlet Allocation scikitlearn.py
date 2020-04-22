
import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)

X = count.fit_transform(df['review'].values)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=10, random_state=123, learning_method='batch')

X_topics = lda.fit_transform(X)

lda.components_.shape

n_top_words = 5
feature_name = count.get_features_names()
for topic_idx, topic in enumerate(lda.components_):
    print('Topic %d:' % (topic_idx + 1))
    print(' '.join([feature_names[i] for i in topic.argsort() \ [:-n_top_words - 1:-1]]))

