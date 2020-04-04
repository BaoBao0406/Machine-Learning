import pandas as pd
df = pd.read_csv('./movie_data.csv')
print(df.loc[0, 'review'][-50:])

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))

print(preprocessor('</a>This :) is :( a test :-)!'))

#df['review'] = df['review'].apply(preprocessor)
    
print(df['review'].head())

