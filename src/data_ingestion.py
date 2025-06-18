import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df = pd.read_csv('./data/raw/sample.csv')

# Separating features and target variable
X = df.drop(columns=['ACCESS_Card'])
y = df['ACCESS_Card']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3','PC4'])
df_pca['ACCESS_Card'] = y.values

df_pca.to_csv(os.path.join('data','processed','sample_pca.csv'), index=False)