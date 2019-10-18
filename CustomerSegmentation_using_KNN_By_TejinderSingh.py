
# coding: utf-8

# # <font color='blue'> Data Science Foundation </font>
# 
# # <font color='#E7135A'> Project Topic- Customer Segmentation using K Means Clustering </font>
# **Project for Module 2- Machine Learning**
# 
# Presenter : Tejinder Singh

# ### Table of Content
# 
# 1. [Data Description](#dd)
# 2. [Importing the dataset and packages](#import)
# 3. [Exploratory Data Analysis](#eda)
# 4. [Finding Clusters with Elbow Method](#elbow)
# 5. [Building K Means model](#Kmeans)

# ### 1. Data Description  <a id='dd'>

# **Retail data** <br>
# This dataset represents the purchase behavior of customers at Spencers’ supermarket in Eastern part of India. 
# 
# **Attributes**  
# - Customer_ID – id of customer
# - AVG_Actual_price_12 – MRP  (in INR) <br>
# - Purchase_Value – Total amount of purchase customer has made (in INR) <br>
# - No_of_Items – Number of items bought   <br>
# - Total_Discount- Discount availed by each customer  <br>
# - MONTH_SINCE_LAST_TRANSACTION – Last month of visit in supermarket  <br>
# 
# In this dataset, 702 rows of measurements of purchases made by various customers have been provided. 
# 
# 
# 
# **Objective** 
# 
# Cluster customers into different segments or groups based on the attributes given using clustering algorithms.
# 

# ### 2. Importing the packages and dataset  <a id='import'>

# In[1]:


# Importing the packages

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Importing the dataset

retail=pd.read_csv("Supermarket_Purchase.csv",index_col='Cust_id')


# In[4]:


retail.columns


# In[5]:


retail.head()


# In[6]:


retail.describe()


# In[7]:


retail.info()


# In[8]:


# Checking the null values in the dataset

retail.isnull().sum()


# ### 3. Exploratory Data Analysis
# #<font color='orange'> <b> Exploraty Data Analysis </b> </font>

# #### Profile Report

# In[9]:


pp.ProfileReport(retail)


# #### Box Plot

# In[10]:


# This shows comparision of items purchased by different retail customers

sns.boxplot(x = 'MONTH_SINCE_LAST_TRANSACTION', y='Purchase_Value', data = retail)


# In[11]:


# This shows variation of MONTH_SINCE_LAST_TRANSACTION with Total_Discount

sns.boxplot(x = 'MONTH_SINCE_LAST_TRANSACTION', y='Total_Discount', data = retail)


# In[35]:


# This shows variation of MONTH_SINCE_LAST_TRANSACTION vis-a-vis No_of_items 

sns.boxplot(x = 'MONTH_SINCE_LAST_TRANSACTION', y='No_of_Items', data = retail)


# #### Correlation plot

# In[12]:


figsize=[10,8]
plt.figure(figsize=figsize)
sns.heatmap(retail.corr(),annot=True)
plt.show()


# ** Strong positive correlation between ** <br>
# No. of Items -- Purchase Value <br>
# Total Discount is more strongly correlated with No. of Items (0.82) than Purchase_Value (0.74) indicating that discounts coupons are offered per unit item instead of total purchase value<br>

# #### Histogram

# In[13]:


## Shows distribution of the variables
retail.hist(figsize=(8,6))
plt.show()


# #### Pairplot

# In[14]:


sns.pairplot(retail)


# # Demystifying Total_Discount 

# In[15]:


# IF you closely observe , you will find value of 'Total_Discount' is higher than 'Purchase_Value' which cannot be the case.
# Also, sometime, 'Total_Discount' is having negative values. Lets analyse these values and try to demystify it.

cust_id = retail.index
purchase_value = retail['Purchase_Value']

total_discount = retail['Total_Discount'];

plt.figure(figsize=(40,10))
plt.plot(cust_id, purchase_value, 'b--', cust_id, total_discount, 'r*-')


# In[25]:


retail.tail(20)


# In[16]:


#scatter plots
plt.plot(retail.Purchase_Value,retail.Total_Discount,'b',color ='red', linewidth = 1.0)
#plt.plot(retail.Purchase_Value,retail.Total_Discount)
plt.xlabel ('Purchase_Value')
plt.ylabel ('Total_Discount')
plt.title ('Purchase_Value vs Total_Discount')
plt.show()


# In[17]:


#scatter plots
plt.plot(retail.No_of_Items,retail.Total_Discount,'b',color ='blue', linewidth = 1.0)
#plt.plot(retail.No_of_Items,retail.Total_Discount)
plt.xlabel ('No_of_Items')
plt.ylabel ('Total_Discount')
plt.title ('No_of_Items vs Total_Discount')
plt.show()


# In[20]:


#scatter plots
plt.plot(retail.AVG_Actual_price_12,retail.Total_Discount/retail.No_of_Items,color ='red', linewidth = 1.0)
#plt.plot(retail.AVG_Actual_price_12,retail.Total_Discount/retail.No_of_Items)
plt.xlabel ('AVG_Actual_price_12')
plt.ylabel ('Discount per unit')
plt.title ('AVG_Actual_price_12 vs Total_Discount')
plt.show()


# In[18]:


# Lets quickly try to establish an equation for Total_Discount as a function of AVG_Actual_price_12,Purchase_Value & No_of_Items 
# using Linear Regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[19]:


# Print multiple statements in same line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[20]:


retail.columns


# In[21]:


#x = retail.drop(['Total_Discount','MONTH_SINCE_LAST_TRANSACTION'], axis = 1, inplace=False)  
x = retail.drop(['Total_Discount'], axis = 1, inplace=False)  
y = retail['Total_Discount']                                  
x.shape
y.shape


# In[22]:


# Split in Train and Test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape
y_train.shape
x_test.shape
y_test.shape


# In[23]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[24]:


predicted = lm.predict(x_test)
predicted.shape


# In[25]:


metrics.mean_squared_error(y_test, predicted)
metrics.mean_absolute_error(y_test,predicted)
metrics.median_absolute_error(y_test, predicted)


# In[26]:


lm.coef_
lm.intercept_
x.columns


# <font color='#E7135A'> <b> With this analysis, it is clear now that... <br> </b> </font>
# 
# <ul>
#     <li> Total_Discount is highly dependent on No_of_items and MONTH_SINCE_LAST_TRANSACTION</li> <br>
#     <li> while it is nominally dependent on  Purchase_Value</li> <br>
#     <li> Also, it is slightly negatively dependent on Avg_Price</li>  <br>
# </ul>

# In[27]:


# Lets consider Negative Values of Total_Discounts as 'Lapsed Discount points', and try to find some pattern in it.
# Add a new column to separate out Negative discount values

retail['discount_lapse'] = np.where(retail['Total_Discount']<0, 'Yes','No')
retail1=pd.pivot_table(retail, index=['MONTH_SINCE_LAST_TRANSACTION'], values=['Total_Discount'],columns=['discount_lapse'],aggfunc=np.sum).fillna(0)
retail1


# In[28]:


retail.drop('discount_lapse',axis=1, inplace=True)


# ** After analysing Total_Discount, Now lets go with clustering exercise **

# In[29]:


retail.columns


# ### 4. Finding Clusters with Elbow Method  <a id='elbow'>

# In[30]:


# Sum Squared Within -> ssw[]
ssw=[]
cluster_range=range(1,10)
for i in cluster_range:
    model=KMeans(n_clusters=i,init="k-means++",n_init=10, max_iter=300, random_state=0)
    model.fit(retail)
    ssw.append(model.inertia_)


# In[31]:


ssw_df=pd.DataFrame({"no. of clusters":cluster_range,"SSW":ssw})
print(ssw_df)


# In[32]:


plt.figure(figsize=(12,7))
plt.plot(cluster_range, ssw, marker = "o",color="cyan")
plt.xlabel("Number of clusters")
plt.ylabel("sum squared within")
plt.title("Elbow method to find optimal number of clusters")
plt.show()


# ** Though Elbow is spotted at No. of clusters=2, while there is no significant change after 4. So for our purpose we will consider optimal No. of Clusters as 3 in order to get more insights **

# ### 5. Building K Means model  <a id='Kmeans'>

# In[33]:


# We'll continue our analysis with n_clusters=3
kmeans=KMeans(n_clusters=3, init="k-means++", n_init=10, random_state = 42)
# Fit the model
k_model=kmeans.fit(retail)


# In[34]:


## It returns the cluster vectors i.e. showing observations belonging which clusters 
clusters=k_model.labels_
clusters


# In[35]:


retail['clusters']=clusters
print(retail.head(10))
print(retail.tail(10))


# In[36]:


## Relative size of each cluster
retail['clusters'].value_counts()


# In[37]:


# Centroid of each clusters
centroid_df = pd.DataFrame(k_model.cluster_centers_, columns=['AVG_Actual_price_12','Purchase_Value','No_of_Items','Total_Discount','MONTH_SINCE_LAST_TRANSACTION']);
centroid_df


# In[38]:


### Visualizing the cluster based on each pair of columns

sns.pairplot(retail, hue='clusters')


# 
# ### Post Segregation Analysis of each Cluster
# <font color='blue'> (1) Cluster 0 - Denoted by BLUE dots </font>
# <font color='orange'> (2) Cluster 1 - Denoted by Orange dots </font>
# <font color='green'> (3) Cluster 2 - Denoted by Green dots <br> </font>
# 
# 
# ### Attributes of each Cluster 
# <font color='blue'> (1) Cluster 0 - Denoted by BLUE dots <br> </font>
# <ol>
#     
#     <li> Total 677 shoppers belong to this cluster. </li> <br>
#     <li> This cluster has highest AVG_Actual_price_12 of 2677 but lowest Purchase_Value=15209 due to lowest No_of_Items (~ 7 items)</li> <br>
#    <li> No. of items purchased is low (~ 7) indicates that don't spend too much in one go i.e. they never over-spend beyond their defined budget limits. That means, No._of_Items is low when Avg_Price is on higher side and vice versa </li> <br>
#       
#     <li>PairGrid of Purchase_Value and Avg_Actual_Price shows that customers of this cluster segment makes Low total purchases but average prices spread across all price range. </li> <br>
# 
# 
# <li> Also, MONTH_SINCE_LAST_TRANSACTION = 5 shows that these are not very frequent buyers and remained INACTIVE most of the time. These can be termed as <b>Quick Bargain Shoppers </b>, who are infrequent and price-sensitive shoppers</li> <br>
# 
# </ol> <br>
# 
# <font color='orange'> (2) Cluster 1 - Denoted by Orange dots <br> </font>
# 
# <ol>
#     <li> Total 19 customers belong to this cluster. </li> <br>
#     <li> This cluster buys items with AVG_Actual_price_12=1619 with No_of_Items=78 and MONTH_SINCE_LAST_TRANSACTION~2 </li> <br>
#     <li> PairGrid of Total_Discount and Average_Actual_Price shows customers of this segment enjoys higher discount points </li> <br>
#     <li> These shoppers can be termed as <b> Variety Shoppers </b> who buy a variety of items, but shop at a lower frequency. </li> <br>
# </ol> <br>
# 
# 
# <font color='green'> (3) Cluster 2 - Denoted by Green dots <br> </font>
# <ol>
#     <li> Just 6 customers belong to this segment/cluster. </li> <br>
#     <li> Can be termed as <b> Loyal Shoppers </b> having higher average spend, highest average item count and frequent buyers. These are the shoppers who are more inclined towards discounts and therefore had availed highest total discount value/points </li> <br>
# 
# </ol> <br>
# 
# 
# <font color='#E7135A'> <b> Recommendation <br> </b> </font>
# Cluster 2 is an ideal shopper segment for this retail supermarket however there are very few customers belong to this segment. Hence, various initiatives should be taken by Spencers’ Supermarket to move shoppers from Cluster 0 to Cluster 1 or 2. Few of the options are listed below...<br>
# <ul>
#     <li> To offer more discount points on items with higher average price values  </li> <br>
#     <li> To offer more incentives if buyers shop more frequently or make repeat purchases</li> <br>
#     <li> As lapsed accumulated discount points are highest in 2nd month of non-usage, it is further recommended not to lapse discount points after 2 months so that buyers will be lured to shop again to make use of those discount points before lapse </li>  <br>
#     <li> Encashment of Discount points should be given on next purchase so that shoppers tend to come back again to use discount points</li> <br>
#     
#     
# </ul>

# 
# ## For any further query, you may drop mail at tejinder.marwah@gmail.com
# 
# 
# ### The End
