#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


MCA = pd.read_csv("Mall_customers.csv")
MCA


# In[3]:


MCA.info()


# In[4]:


MCA.isnull().sum()


# ### Observations
# - There are two null values in Annual_spend and Visit_Frequency
# - Removing null values using mean and median

# In[6]:


median_annual_spend=MCA["Annual_Spend"].median()
mean_annual_spend=MCA["Annual_Spend"].mean()
print("Median of Annual_Spend: ",median_annual_spend)
print("Mean of Annual_Spend: ",mean_annual_spend)


# In[7]:


MCA['Annual_Spend']=MCA['Annual_Spend'].fillna(median_annual_spend)
MCA.isnull().sum()


# ### Observations
# - Now there are no null values in Annual_spend

# In[9]:


median_visit_frequency=MCA["Visit_Frequency"].median()
mean_visit_frequency=MCA["Visit_Frequency"].mean()
print("Median of Visit_Frequency: ",median_visit_frequency)
print("Mean of Visit_Frequency: ",mean_visit_frequency)


# In[10]:


MCA['Visit_Frequency']=MCA['Visit_Frequency'].fillna(median_visit_frequency)
MCA.isnull().sum()


# ### OBSERVATIONS
# - There are no NULL VALUES
# - There are 7 observations (7 different columns)
# - The data types of the columns are int64 and object(6-int64,1-object)

# In[12]:


MCA[MCA.duplicated()]


# ## observation
#  - there is no duplicate values

# In[14]:


print(type(MCA))
print(MCA.shape)


# In[15]:


MCA.dtypes


# In[16]:


print(MCA.head())


# In[17]:


print(MCA.describe())


# In[18]:


print(MCA['Gender'].value_counts())


# In[19]:


cols = MCA.columns
colours =['black', 'green']
sns.heatmap(MCA[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# ## outliers detection
# 

# In[21]:


fig, axes = plt.subplots (2, 1, figsize= (8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in first (top) subplot
sns.boxplot (data=MCA ["Age"], ax=axes [0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title ("Boxplot")
axes[0].set_xlabel("Age Levels")

#plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(MCA["Age"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Age Levels")
axes[1].set_ylabel("Spending Score")
#Adjust layout for better spacing

plt.tight_layout()

#show the plot
plt.show()


# ## Observations
#  - The AGE column has outliers 
#  - it is right-skewed histogram

# In[23]:


fig, axes = plt.subplots (2, 1, figsize= (8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in first (top) subplot
sns.boxplot (data=MCA ["Annual_Spend"], ax=axes [0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title ("Boxplot")
axes[0].set_xlabel("Annual_Spend Levels")

#plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(MCA["Annual_Spend"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Annual_Spend Levels")
axes[1].set_ylabel("Spending Score")
#Adjust layout for better spacing

plt.tight_layout()

#show the plot
plt.show()


# In[24]:


fig, axes = plt.subplots (2, 1, figsize= (8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in first (top) subplot
sns.boxplot (data=MCA ["Visit_Frequency"], ax=axes [0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title ("Boxplot")
axes[0].set_xlabel("Annual_Spend Levels")

#plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(MCA["Visit_Frequency"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Visit_Frequency Levels")
axes[1].set_ylabel("Spending Score")
#Adjust layout for better spacing

plt.tight_layout()

#show the plot
plt.show()


# ### Clustering

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### observation
# - Importing libraries

# In[28]:


MCA = pd.read_csv("Mall_customers.csv")
MCA


# ### observation
# - loading the dataset

# In[30]:


MCA.info()


# ### observation
# - This show the information of dataset

# In[32]:


MCA1 = MCA.iloc[:,1:]
MCA1


# ### observation
# - This will remove the first column from MCA permanently.

# In[34]:


cols = MCA1.columns
cols


# ### observations
# - It will returns the column names of the MCA DataFrame.

# In[36]:


MCA1 = pd.get_dummies(MCA1, drop_first=True)
MCA1


# ### observations
# - MCA1 is now fully numeric with categorical variables converted into dummy variables, making it suitable for clustering. 

# In[38]:


from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()  # Define the scaler

# Assuming MCA1 is a DataFrame
cols = MCA1.columns  # Get updated column names
scaled_MCA_df = pd.DataFrame(scaler.fit_transform(MCA1), columns=cols)  # Apply scaling



# ### observations
# - MCA1 changed after pd.get_dummies(), so cols must be updated

# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_MCA_df = pd.DataFrame(scaler.fit_transform(MCA1),columns = cols)
scaled_MCA_df


# ### observations
# - contains standardized values 

# In[42]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
scaled_MCA_df = imputer.fit_transform(scaled_MCA_df)


# ### observation
# - The code fills missing values in scaled_MCA_df with the mean of each column using SimpleImputer from sklearn.impute.

# In[44]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_MCA_df)


# ### observation
# - The code applies K-Means clustering with 3 clusters 

# In[46]:


clusters_new.labels_


# ### observation
# - Returns an array of cluster labels assigned to each data point in scaled_MCA_df

# In[48]:


set(clusters_new.labels_)


# ### observation
# - returns the unique cluster labels found in the dataset

# In[50]:


MCA['clusterid_new']  = clusters_new.labels_
MCA


# ### observation
# - storing the cluster labels assigned by the K-Means model.

# In[52]:


MCA.sort_values(by = "clusterid_new")


# ### observation
# - grouping similar cluster-labeled data points together.

# In[54]:


MCA = pd.get_dummies(MCA, drop_first=True)  # One-hot encoding
MCA


# ### observation
# - The code converts categorical columns in the MCA dataframe into numbers (0s and 1s) 
# - so that the data can be used in machine learning, while removing one category from each to prevent redundancy.

# In[56]:


MCA.iloc[:,1:].groupby("clusterid_new").mean()


# ### observation
# - The code calculates the average values of all columns (except the first) in MCAto analyze the characteristics of each cluster.

# In[58]:


MCA[MCA['clusterid_new']==0]


# ### observation
# - The code filters the MCA dataframe to show only the rows where the clusterid_new value is 0, displaying data points belonging to cluster 0.

# In[60]:


wcss = []
for i in range(1, 20):

    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_MCA_df)
    #kmeans.fit(Univ)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('number of clusters')
print(wcss)
plt.ylabel('WCSS')
plt.show()


# ### observation
# - The code uses the Elbow Method to determine the optimal number of clusters for K-Means by plotting the Within-Cluster Sum of Squares (WCSS) for cluster counts from 1 to 19.

# In[62]:


#QUALITY OF CLUSTERS IS EXPRESSED IN TERMS OF SILHOUETTE SCORE
from sklearn.metrics import silhouette_score
score = silhouette_score(scaled_MCA_df, \
                         clusters_new.labels_, metric='euclidean')
score


# ### DBSCAN

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Mall_customers.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Drop rows with missing values
df_cleaned = df.dropna().copy()  # Ensure a new copy to avoid warnings

# Selecting numerical features for clustering
features = df_cleaned[['Annual Income (k$)', 'Spending Score (1-100)', 'Annual_Spend', 'Visit_Frequency']].values

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply DBSCAN with adjusted parameters (for ~3 clusters)
dbscan = DBSCAN(eps=1.0, min_samples=8)  # Adjusted eps and min_samples
clusters = dbscan.fit_predict(features_scaled)

# Add cluster labels safely
df_cleaned.loc[:, 'Cluster'] = clusters  # Avoids SettingWithCopyWarning

# Visualizing two features: Annual Income vs Spending Score
plt.figure(figsize=(8, 6))
plt.scatter(df_cleaned['Annual Income (k$)'], df_cleaned['Spending Score (1-100)'], c=clusters, cmap='viridis', marker='o', alpha=0.6)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering of Customers (~3 Clusters)")
plt.colorbar(label="Cluster Label")
plt.show()

# Save the clustered data
df_cleaned.to_csv("dbscan_clusters.csv", index=False)
print("Clustered data saved to 'dbscan_clusters.csv'.")


# In[65]:


from sklearn.metrics import silhouette_score

# Compute silhouette score for DBSCAN
score = silhouette_score(features_scaled, dbscan.labels_, metric='euclidean')
print("Silhouette Score:", score)


# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Mall_customers.csv"  # Update if needed
df = pd.read_csv(file_path)

# Drop missing values
df_cleaned = df.dropna().copy()

# Selecting relevant numerical features
features = df_cleaned[['Annual Income (k$)', 'Spending Score (1-100)', 'Annual_Spend', 'Visit_Frequency']].values

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ✅ Reduce dataset size to 5000 for clustering to prevent MemoryError
df_sample = df_cleaned.sample(n=min(5000, len(df_cleaned)), random_state=42)
features_sample = df_sample[['Annual Income (k$)', 'Spending Score (1-100)', 'Annual_Spend', 'Visit_Frequency']].values
features_sample_scaled = scaler.transform(features_sample)

# Create Dendrogram (on smaller dataset)
plt.figure(figsize=(10, 5))
sch.dendrogram(sch.linkage(features_sample_scaled, method='ward'))
plt.title("Dendrogram for Hierarchical Clustering (Limited to 5000 Samples)")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

# Apply Agglomerative Clustering (on reduced dataset)
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')  # Changed to 3 clusters as requested
df_sample['Cluster'] = hc.fit_predict(features_sample_scaled)

# Save the clustered data
df_sample.to_csv("hierarchical_clusters_sampled.csv", index=False)
print("Clustered data saved to 'hierarchical_clusters_sampled.csv'.")


# In[67]:


from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Mall_customers.csv"  # Update if needed
df = pd.read_csv(file_path)

# Drop missing values
df_cleaned = df.dropna().copy()

# Sample a subset of the data to avoid MemoryError
df_sample = df_cleaned.sample(n=10000, random_state=42)  # Reduce if still running out of memory

# Selecting relevant numerical features for clustering
features = df_sample[['Annual Income (k$)', 'Spending Score (1-100)', 'Annual_Spend', 'Visit_Frequency']].values

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply Hierarchical Clustering with 3 clusters
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df_sample['Cluster'] = hc.fit_predict(features_scaled)

# Compute silhouette score for Hierarchical Clustering
silhouette_avg = silhouette_score(features_scaled, df_sample['Cluster'])
print("Silhouette Score:", silhouette_avg)

# Save the clustered data
df_sample.to_csv("hierarchical_clusters_sampled.csv", index=False)
print("Clustered data saved to 'hierarchical_clusters_sampled.csv'.")


# ### OBSERVATIONS  
#  -Silhouette score for kmean clustering :0.13451900661753186 
#  -Silhouette score for DBscane : 0.46218573149758196  
#  -Silhouette Score for hierarchical: 0.10471098982503971

# In[69]:





# In[70]:


import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN

# Load dataset
df = pd.read_csv("Mall_customers.csv")

# Encode 'Gender'
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Select only relevant columns
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Scale data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Apply DBSCAN Clustering
eps = 0.5  # Adjust based on data
dbscan = DBSCAN(eps=eps, min_samples=5)  # min_samples can also be tuned
df['Cluster'] = dbscan.fit_predict(scaled_features)

# Save the model and scaler
joblib.dump(dbscan, 'dbscan_modell.pkl')
joblib.dump(scaler, 'scalerr.pkl')

print("DBSCAN Model & Scaler Saved Successfully!")


# In[71]:


import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained DBSCAN model and scaler
dbscan = joblib.load("dbscan_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Mall Customer Segmentation App (DBSCAN) 🛍️")
st.markdown("Predict the customer cluster using DBSCAN based on Age, Gender, Income & Spending Score.")

# Sidebar Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 25)
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

# Convert Gender to numerical value
gender_encoded = 1 if gender == "Male" else 0

# Prepare input data
input_data = np.array([[gender_encoded, age, income, spending_score]])
scaled_input = scaler.transform(input_data)  # Scale input

# Predict Cluster
cluster = dbscan.fit_predict(scaled_input)[0]

# Display Output
st.subheader("Prediction Result 🎯")
if cluster == -1:
    st.write("The customer is classified as **Noise** (not assigned to any cluster).")
else:
    st.write(f"The customer belongs to **Cluster {cluster}**")

# Add explanations
st.markdown("""
### Cluster Meanings (Example Interpretation):
- **Cluster 0**: Low spenders, low income
- **Cluster 1**: Moderate spenders, middle-class
- **Cluster 2**: High spenders, high income

(DBSCAN does not always form the same number of clusters; interpretation may vary.)
""")

