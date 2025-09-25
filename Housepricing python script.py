# %%
import sys
assert sys.version_info>=(3,5)

import sklearn
assert sklearn.__version__>="0.20"

import numpy as np
import os #for os operations

# to plot pretty graphs

import matplotlib as mpl
# displays plots within the cell
%matplotlib inline
import matplotlib.pyplot as plt
# setting up graph parameters
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)

#setting up project dir to save images and files
# "." dot indicates to current folder
PROJECT_ROOT_DIR="."
CHAPTER_ID ="end_to_end_project"
#joining paths 
IMAGE_PATH=os.path.join(PROJECT_ROOT_DIR,"images",CHAPTER_ID)
os.makedirs(IMAGE_PATH,exist_ok=True)

# function to save plotted graphs in high res and as png 
def save_fig(fig_id,tight_layout=True,fig_extension="png",resolution=300):
    path=os.path.join(IMAGE_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %%
# getting/fetching the data from url

import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    #extracting csv file from tarfile 
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# %%
fetch_housing_data()
#fetching data 


# %%
import pandas as pd
#loading csv file into a df using oandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# %%
housing=load_housing_data()
housing.head()

# %%
housing.info()

# %%
housing['ocean_proximity'].value_counts()

# %%
housing.describe()

# %%
#generating hsitogram for each numerical column
housing.hist(bins=50,figsize=(20,15))


plt.show()
save_fig("attribute_histogram_plots")

# %% [markdown]
# # the inferences that can be made from the above hsitograms are :
# 1. Median income is capped between 0.5 and 15 any value lower than 0.5 becomes 0.5 and greater than 15 becomes 15
# 2. The house median age and median house value were also capped, the later is serious problem as it is the target value, our machine larning algo may learn that the values cant go beyond that limit, check up with the team if its not a problem.
# 3. If they tell you that they need precise predictiion even beyond 500,000 then there are two options left with us :
#      a) collext proper labels for those districts whose labels were capped.
#      b) Remove those districts from the training set (and also from the test set, since your
#         system should not be evaluated poorly if it predicts values beyond $500,000).
# 4. These attributes have very different scales.
# 5. Finally, many histograms are tail wavy, they extend much farther to the right of the median
#    than to teh left.This may make it a bit harder for machine learning algorithm to detect
#    patterns. we will try to eliminate this skewness to get more bell shaped distribution.
#    

# %%


# %%
np.random.seed(42)

# %%
def split_train_test(data,test_ratio):
    #len(data) returns number of rows
    #np.random.perm.. returns an array of random indices from 0 to len(data)-1
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]    

# %%
train_set,test_set=split_train_test(housing,0.2)
len(train_set)

# %%
len(test_set)

# %%
#the above method used is traditional method but contains flaws : 
# it will fail when the new data will be added
# better method is using hashmap using crc32 algo
# but for the hash map thing we need an identifier and we dont have an identifier column hence simplest method is using row index

# %%
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# there are seveeral other methods too for splitting the data using hashing

# %%
# splitting using functions of sklearn 
# features provided : - the functions works same as the random seed generator one 
# - additional features it provides is, it can split two datasets by taking them as parameter
#   (with identical number of rows) and it will split it using the same indexes
#   (example we have different dataframe for labels)
# using this method is fine if our data set is very huge but fails when the dataset is small like in our case,
# using this method can cause sampling bias
# suppose a survey company decides to call 1000 people to ask quesitions it wont call randomly , suppose us population 
# got 52 % male and 48 % female then the sample of 1000 must be the representative of the whole popultion hence the
# selected 1000 people will contain 480 females and 520 males this is called stratified sampling : 
# the population is divided into homogenous subgroups called strata and the right number of instances is sampled
# from each stratum ( homogenous subgroups ) is selected to ensure the test set is representative of 
# whole data set
# if any attribute has continuous numerical values then it will form too many stratas ( homogenous subgrouos ) hence we categorize this data 
# and now the stratas will be less and the split of train and test data is easier and now the test data can be made representative of the whole
# data by using stratified shuffling


# %%
housing['income_cat']=pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])

# %%
housing['income_cat'].value_counts()

# %%
housing['income_cat'].hist()
plt.show()

# %%
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
    # splitting accorrding to income category ensuring the ratio is maintained.

# %%
# lets see if the proportions we wanted to maintain were actually maintained or not

strat_test_set['income_cat'].value_counts()/len(strat_test_set)

# %% [markdown]
# 

# %%
#checking proportions in parent dataset - housing
housing['income_cat'].value_counts()/len(housing)

# %% [markdown]
# now we can see the split by the stratified fun is almost identical  in proportions to the housing dataset

# %%
#Now you should remove the income_cat attribute so the data is back to its original state:

for set_ in (strat_train_set, strat_test_set): 
    set_.drop("income_cat", axis=1, inplace=True)

# %%
housing.head()



# %%
test_set.head()
# housing['median_house_value'].value_counts()

# %%
strat_test_set.head()
# we will continue with the strat sets only

# %%
# WHAT WE HAVE LEARNED SO FAR (QUICK RECAP):

- after loading the csv  file in a dataframe we must look at the dataset using methods like
  info(),describe(), for any column with repeating values we can form categories and see how many   instances fall in that category using value_count(), this automatically forms the catogaries    
  and give the number of data instances in each catogary NOTE : null values are ignored durin the   mean, count and other summary calcullation of numerical attributes.
  
- After we must plot all the numeric features and analyze them, we see skewness, we check if any    features of the dataset are capped between certain values or not, check if capping creates   
  an issue, checking if the attributes have very differet scales which need to be fix, checking 
  if the histograms are tail heavy or not which can create an issue in detecting patterns while     training the model w will try to transform these attributes to have more bell shaped 
  distributions.

  

# %%
CREATING A TEST SET :

THERE ARE SEVERAL METHODS TO CREATE A TEST SET :
1. Using conventional coding, extracting the indices randomly using test ratio and then returning the data.

2.Best method is using hashing which even works if new data is being added to the data set it will split the data into test and training set in most effective way.For example, you could compute a hash of each instance’s identifier and put that instance in the test set if the hash is lower or equal to 20% of the maximum hash value. This ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset. There are several methods of hashing too which are stated in book with examples, but hashing requires indentifiers and using rows as identifier can create issues hence we need to make an identifier which stays constant during the whole life cycle and code.

3.Other method is using sklearn functions which works same as the 1st method but it can split muultiple datasets at the same time suppose we need to split the dataset which contains the label of the instaces, also it provides stratified split which maintains the proportions of a strat stated while splitting the dataset

# we use cut function to make the continous numerical data in to categorial form.(binning).


# %%
DISCOVERING AND VISUALSING THE DATA TO GAIN INSIGHTS :
- putting the test aside now we will work only with the train set
- if the training set is, we may want to sample an exploration set so that the original test sset   data donot get disturbed if any mistake made.


# %%
# creating copy
housing_copy=housing.copy()
housing=strat_train_set.copy()

# %%
# visualizing the geaographical data
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()

# %%
# setting up alpha=0.1 helps us to see where the datapoints are densely present the
# region appears brighter than the others
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# %%
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,
    label="Population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)

plt.legend()
plt.show()
save_fig("housing_prices_scatterplot")

# %%
explaining the code :

s = size of each dot , = population/100 ==> bigger the population bigger the dot
label = labels the graph

c = parameter sets the color = "median_house_value" , each house is colored on the basis of the oassed value , higher prices -> different colors based on color map

cmap= color map used to represent values in color = plt.get_cmap("jet") : jet ranges from blue → green → yellow → red
(low → high values) , blue = low house values , red =  high house values

colorbar=True
Adds a color scale on the side of the plot
Helps user know what colors represent what price ranges

# %%
# LOOKING FOR CORRELATIONS 
# Since the dataset is not too large, you can easily compute the 
# standard correlation coefficient (also called Pearson’s r) between every 
# pair of attributes using the corr() method:

corr_matrix=housing.corr(numeric_only=True)
corr_matrix['median_house_value'].sort_values(ascending=True)

# %%
Another way to check for correlation between attributes is to use Pandas’ scatter_matrix function, which plots every numerical attribute against every other numerical attribute. Since there are now 11 numerical attributes, you would get 11? = 121 plots, which would not fit on a page, so let’s just focus on a few promising attributes that seem most correlated with the median housing value (Figure 2-15):

# %%
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age" ]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()

# %%
# after analyzing the correlation matrix it is evident that the most promising attribut to predict the 
# median house value is median income so lets analyze and zoom in on their correlation
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
plt.show()

# %% [markdown]
# The plot reveals following things :
# 1. The correlattion in indeed very strong, less dispersed plot
# 2. Price cap at 500k which we noticed earlier (horizontal straight line)
# 3. other straight lines possibly due to :
#     - rounding
#     - tax/legal pricing
#     - survey recording limitations
# These visible lines may confuse model and the model could overfit these quirks rathar than generalize real world trends.

# %%
housing.plot(
    kind="scatter", 
    x="median_income", 
    y="median_house_value", 
    alpha=0.4,
    s=housing["population"] / 100,  # bubble size
    label="population",
    figsize=(10, 7),
    c="median_house_value",  # color by price
    cmap=plt.get_cmap("jet"), 
    colorbar=True,
)
plt.legend()
plt.show()

# %%
✅ Key Insights from Data Exploration:
1.Identified data quirks (e.g., price caps at $500,000) that may need to be cleaned before training ML models.

2.Found strong correlations between features and the target attribute (median_house_value), especially with median_income.

3.Noticed tail-heavy (skewed) distributions in some attributes (e.g., median_income, total_rooms).
  which we may want to transform.

4.Considered logarithmic transformations for skewed features to normalize distributions.

5.Emphasized the importance of visual exploration using scatter plots, correlation matrices, and histograms.

6.Highlighted that data preprocessing steps will vary by project, but these general ideas apply broadly.


# %%
EXPERIMENTING WITH ATTRIBUTE COMBINATION : (feature engineering)

# %%
housing.head()

# %%
housing['rooms_per_household']=housing['total_rooms']/housing['households']
housing['population_per_household']=housing['population']/housing["households"]
housing['bedrooms_per_room']=housing['total_bedrooms']/housing['total_rooms']

# %%

corr_matrix=housing.corr(numeric_only=True)
corr_matrix['median_house_value'].sort_values(ascending=False)

# %%
Its time to prepare the data for your Machine Learning algorithms. Instead of just doing
this manually, you should write functions to do that, for several good reasons:

- the transformation will still be applied even if the dataset is changed or updated.

- by writing these many functions we might build our own library whichh we 
  can use later in any other project

- You can use these functions in your live system to transform the new data before feeding
  it to your algorithms.

- As we are writing functions hence we can try many transformations in the data, but it is not
  possible manually as once changed manually it will be tideous task to get back to the same dataset.


# %%
lets revert to data cleaning step first :
- we will create copy of the strat test set again and drop the labels that is median house value from the dataset copy and store it in housing_labels dataset.
- we are separating the predictor that is features froom the labels or target values as we dont want same changes to be applied to both.

# %%
housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set['median_house_value'].copy()
housing.info()

# %%
DATA CLEANING :

# %%
# as we know moat learning algos cant  work with missing values hence we need to handle them in some way :
# three methods :
# 1. get rid of districts - dropna
# 2. get rid of whole attribute - drop function
# 3. set the values to some value (zero,median ir the mean etc) using fillna


# %%
# if we use the option 3 then we neeed to dave the value somewhere as we need to replace the missing values in test set as well
# and same goes when working with the new or updated data.
# another method is using sklearn's imputer classs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# since the median can be computed only for the numerical values then we will create a copy of train seet with all numerical values
# skipping ocean proximity attribute
# this is applied on the all columns hence the missing values of all the columns will 
# be replaced by their median values this is the reason we need to remove
# non numeriical columns
housing_num=housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
# the above token trains the imputer , it calculates and stores the median value of each column.
# these medians will later be used to fill in missing values.
# initially imputer saves them in .statistics_


# %%
imputer.statistics_


# %%
#checking if its same as manually computing the values
housing_num.median().values

# %%
# this actually replaces the missing values with the median and returns a numpy array as all the outputs of sklearn gives numpy array as outputs as the calcualtions using numpys are faster

x= imputer.transform(housing_num)
x

# %%
housing_tr=pd.DataFrame(x,columns=housing_num.columns)
housing_tr

# %% [markdown]
# AS OF NOW WE HAVE HANDLED THE MISSING VALUES.
# 

# %% [markdown]
# NOW, HANDLING TEXT AND CATEGORIAL ATTRIBUTES :

# %%
housing_cat=housing[['ocean_proximity']]
housing_cat.head()

# %%
# most ml alf=gorithms prefer to work with numbers so lets convert this to numerical by encoding:
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
# the above token returns an nd array and then it is stored in housing cat encoded 
housing_cat_encoded[:10]
# type(housing_cat_encoded)
#Problem:
# ML models treat numbers like ordered values.
# So it might assume:

# INLAND (1) is closer to ISLAND (2) than to NEAR OCEAN (4) — which makes no sense here!

# %%
# solution to above problem is one hot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# the output here is sparse matrix which contains zero and 1 for he presence of the attributes like studied befoe
# this one hot encoding forms new attributes and these are called dummy attributes if one attribute is present
# suppose one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise) 
# sparse matrix only stores the location and value of non zero attribute as shown below
# [
#  (0, 2)  1.0,
#  (1, 1)  1.0,
#  (2, 3)  1.0,
#  ...
# ] sparse matrix

# %%
#converting to array
housing_cat_1hot.toarray()


# %%
cat_encoder.categories_

# %%
# all the built in functions of sklearn we used for transformation are called transformer but we can also 
# make our own custom transformers
# we use these transformers mainly in data preprocessing pipelines
# and what all we have done with the data till now before training the mdoel is part of the data preprocessing pipeline

# %%
WHAT WE HAVE DONE SO FAR ?
1. splitting of data into test and train set and setting the test dataset aside.
2. separating labels from the training set
3. checking corrrelations and distributions
4. creating new attribute / building new features and again checking correlations
5. handling missing values using transformers
6. handling categorial data using one hot encoding transformer
- there is more to go before actually training the dataset
- in transformers, first we fit or train the transformer using fit()
  and then transform the dataset using transform()

# %%
CUSTOM TRANSFORMERS :

Reason to build :
1. Create new festures
2. Applly custom cleanup logic
3. do somethig that is domain specific that built ins dont handle

A transformer is just a Python class that:
 - Has a .fit() method → learns something (or just returns self)
 - Has a .transform() method → applies the transformation
 - and fit_transform() , wee can get this for free by adding TransformMixin.
 - Optionally: inherits from helper base classes
Also if we add BaseEstimator to the class as baseclass:
 - we will get two extra methods get_params and set_params as free
 - these methods are useful for hyper parameter tuning.


# %%
EXAMPLE OF CUSTOM TRANSFORMER , THAT ADDS THE COMBINED ATTRIBUTES AS WE DISCUSSED EARLIER :

# %%
# from sklearn import BaseEstimator, TransformerMixin
# rooms_ix,bedrooms_ix,population_ix,households_ix = 3,4,5,6

# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def __init__(self,add_bedrooms_per_room=True):
#         sef.add_bedrooms_per_room = add_bedrooms_per_room
#     def fit(self,X,y=None):
#         return self
#     def transform(self,X,y=None):
#         rooms_per_household=X[:,rooms_ix]/X[:,households_ix]
#         population_per_household=X[:,population_ix]/X[:,households_ix]
#         if self.add_bedrooms_per_room :
#             bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
#             return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
#         else:
#             return np.c_[X,rooms_per_household,population_per_household]
# attr_adder=Combined

# %%
HANDS ON ML 3 FROM HERE ONWARDS :

# %% [markdown]
# *Feature Scaling and Transformation*

# %%
# - Machine learning algos dont perform well when the input numerical attribute have very different scales.
# - the algo becomes biased towards the higher scale values like in our case it will be no. of rooms 
#   which is less important than the median income.
# - to tackle this we use feature scaling 

# TIP :
# - As with all esitmators we only fit the scalers to training data only.
# - we never use fit( ) or fit.transform for any other set than the training set by doing this we will be letting the model peek into the test or validation set.
# - once we have trained the scaler on the training set now we can apply transform everywhere(test , validation or test dev set)
# - note that the training set values will be scaled to the specified range only if the new data contains outlierrs then they will be scaled out of range to tackle this,
#   just set the clip hyperparameter to True


# %%
housing_num


# %%
# There are two common ways to get all attributes to have the same scale: min
# max scaling and standardization :
# 1. MinMax scaler :
# - capping from 0 to 1.
# - feature_range hyperparameter to change  the range
# - NN prefers ero mean range (-1,1)
from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled=min_max_scaler.fit_transform(housing_num)

# 2.Standard Scaler :
# - Unlike mean it doesnt restrict value to a specific range.
# - less affected by outliers.

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled=std_scaler.fit_transform(housing_num)




# %%
x = pd.DataFrame(housing_num_std_scaled, columns=housing_num.columns.tolist())
x

# %%
# # dealing with heavy tail distributions :
# reason : when the features distribution is heavy tailed if we use scalers then both the scalers will squash
#          most values (around peak) around the zerom which all the models generally dont like.So it is prefered
#          to turn the distribution in roughly symmetric distribution so that models can learn easily.
#          The transformation converts the skewed distribution into  a bell shaped or gaussian 
#          which is preffered by the models to elarn effectively

#  ==== we should always transform before scaling the feature


# 1st approach :
# - replacing the feature with its logarithmic, square root, or raise the power between 0 and 1.
# - positive value wiht right skew == square root.
#   moderate right skew == power tramsform (0<p<1)
#   very heavy right tail == logarithmic  

# 2nd approach :
# - using bucketing by percentiles 
# - bucketing is breaking the continuous numeric features into discrete buckets.
# - but bucketing here should be numeric like using percentiles :  bucket 0 - lowest 20 percentile,
#   bucket 1 - 20 to 40 , ..... bucket 4 - top 20%
# - after bucketing replace the original values with the bucket index.
# - now the transformed feature is skew handled , doesnt care about outliers ,uniformly distributed ,moel ready no scaling needed.
# - for example income_cat in satratified sampling.
# - but bucket misses some details from the original data , works better when precise numeric relationshiips dont matter.


# %%
plt.hist(housing['housing_median_age'],bins=50)
plt.show()

# %%
# - As we can see the above graph has multiple clear peaks called modes this dsitributin is called
# multimodal distribution. Bucketizing these features can also be helpful but this time we bucketize
# them as categories not numbers so we dont say bucket 2 > bucket 1.

# - this means that after bucketizing them as categories we encoode them using onehotencoder when the
# number sof buckets is not huge.

# -This approach will allow the regression model to
#  more easily learn different rules for different ranges of this feature value. 
  
#   @Why This Works Better:
#  - Captures non-linear effects (like sudden drops or bumps in price)
#  - Allows the model to fit different patterns for each range
#  - One-hot encoding avoids incorrect assumptions about order/distance

# %% [markdown]
# Another approach for handling the multimodal distribution is using rbf (Radial Basis Function) :
# - In this approach we take few important peaks say 35 then we make new features from them.
# - The feature is : closeness to the mode (35).
# - rbf = exp(–γ(x – 35)²). and the gamma here is hyperparameter which determines how quickly the
#   similarity decays on increasing distance from the mode
# - This closeness or farness from the peak is measured using rbf function which deacreases from
#   1 to 0 on increasing distance from the mode.
# - This mode doesnt force boundaries like bucketing , and the curve formed is similar to gaussian    bell shape.
# 
# This approach is implemented using the rbf_kernel from sklearn :
# from sklearn.metrics.pairwise import rbf_kernel
# age_simil_35
# 

# %%
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]],[[35],[34]],gamma=0.1)


# %%
pd.DataFrame(age_simil_35)

# %%
ages = np.linspace(housing["housing_median_age"].min(),
                   housing["housing_median_age"].max(),
                   500).reshape(-1, 1)
gamma1 = 0.1
gamma2 = 0.03
rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Housing median age")
ax1.set_ylabel("Number of districts")
ax1.hist(housing["housing_median_age"], bins=50)

ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
color = "blue"
ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel("Age similarity", color=color)

plt.legend(loc="upper left")
save_fig("age_similarity_plot")
plt.show()

# %%
Till now we have transformed the attributes but the target values need to be transformed sometimes
as in our case, as it can also have skewed distribution,heavy tail,large numerical range.
- hence we need to transform the target value as well.
- procedure is as Below in the code :


# %%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#scaling the target value
target_scaler = StandardScaler()

#converting the series housing_labels to a dataframe as scaler need 2d input
scaled_labels=target_scaler.fit_transform(housing_labels.to_frame())

#choosing the model
model=LinearRegression()
# fitting the data , median income only , in the model with labels 
model.fit(housing[["median_income"]],scaled_labels)

some_new_data=  housing[["median_income"]].iloc[:5] #pretendin this to be new data
# pedicting values, the values will be in the scaled form as we scaled the values before taking the,
# to train the model as labels, no now we need to covnert them to standrd form.
# this reversal is possible because most of the transformations are reversible in sklearn
# 
scaled_predictions = model.predict(some_new_data)

# reverse transformation
predictions = target_scaler.inverse_transform(scaled_predictions)
predictions

# %%
# There is a shortcut to above method that is using the transformed regression
from sklearn.compose import TransformedTargetRegressor
model= TransformedTargetRegressor(LinearRegression(),transformer = StandardScaler())
model.fit(housing[["median_income"]],housing_labels)
predictions = model.predict(some_new_data)

# %%
CUSTOM TRANSFORMERS :

# %%
# sklearn comes out to be really helpful while creating the custom transformers.
# It provides a wrapper FunctionTransformer which takes any function and turns it into a scikit Learn style 
# like Transformer and alloows it to be used in the data pipelines like any other of its tr. it also 
# provides the hyperparameter inverse transform for inversing the transformation.
# input and output as np array
from sklearn.preprocessing import FunctionTransformer
import numpy as np
log_transformer = FunctionTransformer(np.log,inverse_func = np.exp)
log_pop = log_transformer.transform(housing[["population"]])

plt.hist(log_pop,bins=50)
plt.show()


# %%
log_pop


# %%
log_transformer.inverse_func(log_pop)

# %%
# using rbf kernel as a function transformer 
rbf_transformer=FunctionTransformer(rbf_kernel,kw_args=dict(Y=[[35]],gamma = 0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# %%
# adding a new columns measuring geographic similarity between each dist and san francisco
# rbf kernel doesnt treat the features separately, if we pass it an array with two features
# then it will calculate the 2d distance between these features.
sf_coords = 37.7749,-122.41
sf_transformer = FunctionTransformer(rbf_kernel,kw_args = dict(Y=[sf_coords],gamma =0.1))
# using dict as the kwargs takes dictionary as inputs to use multiple aprameters at the same time
sf_simil = sf_transformer.transform(housing[['latitude','longitude']])
sf_simil

# %%
# custom transformer are also useful to combine the featrues such as calulating ratios or adding other
# mathematical operation impplied using lamba function
ratio_transformer = FunctionTransformer(lambda X : X [:,[0]]/ X[:,[1]])
ratio_transformer.transform(np.array([[1.,2.],[3.,4.]]))

# %%
x_train

# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self,with_mean =True):
        self.with_mean = with_mean

    def fit(self,X,y=None):
        X=check_array(X) # validates if input data is or correct type 
        self.mean_ = X.mean(axis= 0)
        self.scale_ = X.std(axis =0)
        self.n_features_in_ = X.shape[1] # how many featues will be there will fitting the training data / validates if the data is fitted in the mdodel
        return self

    def transform(self,X):
        check_is_fitted(self) # checks if the model is fitted or not as we dont call transform method without fitting the model
        X=check_array(X)
        assert self.n_features_in_ == X.shape[1] # asserts if the number of features is equal to the number of columns in the input dataset as shape returns a 2-tuple and shape[1] gives the 2nd element of the tuple.
        if self.with_mean :
            X=X - self.mean_
        return X/self.scale_

# %% [markdown]
# ***Few things to note:***
# 1. sklearn.utils.validation provides vaious methods to validate the inputs.
# 2. SKlearn pipelines require the fit method to have two arguments X and y which is why we need the y = None argument even though we dont use y.
# 3. all sklearn estimators set n_features_in in the fit() method and they ensure that the data passed to transform or predict has this number of features.
# 4. the fit method must return self it is expected in skelarn
# 5. this implementation is not complete , all th  e estimators must pass get_features_name_out() in the fit() method if we are passing a dataframe, as well as inverse_transform() method whenn their transformation can be reversed.
# 
# ***Why do we use these methods and pass them as arguments in the fit method () ?***
# 
# - get_features_name_out() : while using the pipelines and columntransformers these methods allows us to trace the features names all the way to model.
# - with_mean is a hyper parameter to subtract mean while feature scaling.
# 
#  these are the attributes of class or methods of class hence these are not passed as arguments in the fit method.

# %%
The attributes ending with _ are called learned attributes. These are the attributes which the estimators calculates or computes during training or which we call fitting typically inside the .fit() method. They are saved as instance variables.

# %% [markdown]
# *A custom transformer can use other estimators (anything that is used to compute or return something) in its implementation. The example below shows the usage of KMeans in the fit() method to identify the main clusters in the training data and then uses rbf kernel to measure how much similar each sample is to each cluster centre :*

# %%

from sklearn.cluster import KMeans
class ClusterSimilarity(BaseEstimator,TransformerMixin):
    def __init__(self, n_clusters=10,gamma=0.1,random_state = None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gamma = gamma
    def fit(self,X,y=None,sample_weight = None):
        self.kmeans_ = KMeans(self.n_clusters,random_state=self.random_state)
        self.kmeans_.fit(X,sample_weight = sample_weight)
        return self
    def transform(self,X):
        return rbf_kernel(X,self.kmeans_.cluster_centers_,gamma = self.gamma)
        
    def get_feature_names_out(self,names = None):
        return [f"Cluster {i} similarity" for i in range (self.n_clusters)]

cluster_simil = ClusterSimilarity(n_clusters =5,gamma = 1.,random_state =42)
similarities = cluster_simil.fit_transform(housing[["latitude","longitude"]],sample_weight = housing_labels)

similarities_df=pd.DataFrame(similarities)
print(similarities_df)
print()

# the dimensions of the returned dataframe is decide by the number of features in the input data and the number of samples in it.
# kmeans clustering algo treats the number of features,n (say n= 2) as a point in n-d space and the cluster centroid also have the representation in the nd
# space coordinates and then the distance between these is calcualted to make a single data point of the similarity dataframe.

# %% [markdown]
# explanation of the code above :
# - KMeans is the algorithm used to find the clusters in the given data.
# - n_clusters = 10 default number of clusters for kmeans.
# - gamma for the rbf.
# - random state = random seed for reproducibility. we are using this because the kmeans is
#   stoichastic algo which depends on the randomness hence we need to seed it.
# - inside the fit method we make a learned attribute that is also an object of the kmeans class.
# - we fit the dataset x in the kmeans algorithm not the clustersimilarity class.
# - the fit method returns self as expected by sklearn.
# - inside the transform method we only have one argument that is data set.
# - rbf kernel takes the data set and the cluster centers (kmeans_.cluster_centers_) to measure
#   similarity between each data point or instance and each cluster center.
# - get feature names out method is used to reutrn the names of the features (output) on which we
#   have conducted any sort of operations.
# - then we create an object using the clustersimilarity class by specifyin the parameters random
#   state ,n_clusters , gamma.
# - after creating the object we apply method fit and transform by passing the required arguments
#   for both fit and transform method and store the result in similarity variable. The array is
#   returned as output we the convert that in the dataframe and print.
#   

# %% [markdown]
# # ***TRANSFORMATION PIPELINES :***

# %% [markdown]
# - As we can see the data goes through a number of transformations during preprocessing and these 
#   steps need to be executed in right order like transformation befoe scaling the feature.
# - Fortunately there exists a pipeline class with pipeline constructor in sklearn to do this.
# - The constructor takes a list of name of the estimator and the estimator.
# - The esitmator must be transformers, they must have the fit_transform method except for the last
#   one, which can be anything : a transformer, a predictor or any other type of estimator.

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])
num_pipeline

# %% [markdown]
# - *If we dont want to name the pipeline we can use make_pipeline() function instead.*
#   
# - *It takes transformers as positional arguments and creates a Pipeline using the names of the
#   transformers’ classes, in lowercase and without underscores. (e.g., simpleimputer).*
# 
# - *If multiple transformers have same name then an index is added at then end of their name to
#   differentiate.*
# 
# - *When we call the pipelines' fit() method then the fit_tranform() method of all the transformers
#   are called sequentially and then the output of the following transformers is passed to the
#   leading one unntil it reaches the last transformer or estimator for which it only calls the
#   fit() methhod.*
# 
# - *The pipeline exposes the same methods as the final estimator specified in the pipeline
#    constructor. In pur case below the final estimator is a transformer hence the pipeline will
#    behave like a transformer*
# 
# - *If you call the pipeline’s transform() method, it will sequentially apply all the
#   transformations to the data.*
# 
# - *If the last estimator is a predictor then the pipeline will behave like a prdictor. Calling it
#   would sequentially apply all the transformations to the data and pass the result to the
#   predictor’s predict() method*

# %%
from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
housing_num # the training dataset containing only numerical value/ data instances not ocen proximity


# %%
housing_num_prepared = num_pipeline.fit_transform(housing_num) #applying transformations to the training datatest
housing_num_prepared[:2].round(2)

# %%
# if we want to recover a nice dataframe we can use get the feature names out method of the pipeline
df_housing_num_prepared = pd.DataFrame(housing_num_prepared,columns = num_pipeline.get_feature_names_out(),index = housing_num.index)
df_housing_num_prepared

# %%
# Pipelines are super flexible and come with useful ways to access and manipulate the individual steps. 
# Pipeline supports indexing like lists :
num_pipeline[0]

# %%
# using .steps attribute
num_pipeline.steps


# %%
# accessing the name and the estimator
step_name,estimator = num_pipeline.steps[0] #unpacks the tuple
print(step_name)
print(estimator)

# %%
# using the .named_steps
num_pipeline.named_steps["simpleimputer"]
# Returns the SimpleImputer instance directly
print(num_pipeline.named_steps["simpleimputer"].strategy) # ret

# %%


# %% [markdown]
# - So far we have dealt with the numerical and categorial data separately. It'd be more convenient to have a transformer which can deal with all the columns, applying the appropriate transformations to the appropriate columns.
# 
# - For this we have ColumnTransformer class, used in the example below.
# - The columnTransformer class cinstructor takes the list of 3-tuples as input which contains :
#   name  , actual name of the transformer , attribute list on which it is going to be applied.

# %%
from sklearn.compose import ColumnTransformer
num_attributes =  num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
 "total_bedrooms", "population", "households", "median_income"]
cat_attributes = ['ocean_proximity']

cat_pipeline = make_pipeline(
    SimpleImputer(strategy = "most_frequent"),
    OneHotEncoder(handle_unknown = "ignore"))

preprocessing=ColumnTransformer(
    [
        ("num",num_pipeline,num_attributes),
        ("cat",cat_pipeline,cat_attributes)
    ]
)

# %% [markdown]
# - The above method is not convenient as we need to lsit all the attributes.
#   
# - Sklean provides make_columns_selector function which takes datatype as the input and selcts the
#   features automatically and we do not need to specify the attribute list.
# 
# - And if we dont care about the names of the pipeline then we can use make_column_transformer just like make pipeline in which we dont need to specify the name we want ot give.

# %%
from sklearn.compose import make_column_transformer, make_column_selector
preprocessing = make_column_transformer(
    (num_pipeline,make_column_selector(dtype_include =np.number)),
    (cat_pipeline,make_column_selector(dtype_include = object))
)

# %% [markdown]
# Now we are ready to apply this pipeline to the housing data :

# %%
housing_prepared=preprocessing.fit_transform(housing)

# %% [markdown]
# - *Now we have our pipeline ready to preprocess the data , which takes the entire training data and applies each transformer to appropriate columns and the concatenates the transformed columns horizontally as the transformer must not change the no. of rows of the dataset.*
#   
# - *once again the output we have is in form of the numpy array but we can get the columns name from the get_features_name_out() methods and we can pack the data into a ncie DataFrame.* 

# %%
df_housing_prepared = pd.DataFrame(housing_prepared,columns = preprocessing.get_feature_names_out(),index = housing.index)
df_housing_prepared

# %% [markdown]
# Now we would want to create a single pipeline to perform all the tasks that we have performed till now, so lets recap what our pipeline will do :
# 
# 1. Missing values in the data will be imputed by simpleimputer using different strategies as most ml algo dont prefer missing values.
#    
# 2. The categorial data will be OneHotEncoded as most ml algos onlly accepts numerical data
# 
# 3. A few ratio features will be added like bedrooms_ratio , rooms_per_house and people_per_house. Hopefully these will better correlate with the median house value.
# 
# 4. A few cluster similarity features will also be added. These will likely be more useful than the longitude and the latitude.
# 
# 5. Features with long tails will be replaced by their logarithms, as most models prefer gaussian distribution or symmitrical data distribution.
# 
# 6. All numerical features will be standardized, as most ML algos prefer when the features have toughly the same scale.
# 
# The code for the pipeline is given below :

# %%
# def column_ratio(X):
#     return X[:,[0]]/X[:,[1]]

# def ratio_name(FunctionTransformer,feature_names_in):
#     return ["ratio"]
# def ratio_pipeline():
#     return make_pipeline(
#         SimpleImputer(strategy = "median"),
#         FunctionTransformer(column_ratio,feature_names_out = ratio_name),
#         StandardScaler()
#     )

# log_pipeline = make_pipeline(
#     SimpleImputer(strategy="median"),
#     FunctionTransformer(np.log,feature_names_out="one-to-one"),
#     StandardScaler()
# )

# cluster_simil=ClusterSimilarity(n_clusters = 10 , gamma = 1. , random_state = 42)
# default_num_pipeline = make_pipeline(
#     SimpleImputer(strategy= "median"),
#     StandardScaler()
# )

# preprocessing = ColumnTransformer([

#     ("bedrooms",ratio_pipeline(),["total_bedrooms","total_rooms"]),
#     ("rooms_per_house",ratio_pipeline(),["total_rooms","households"]),
#     ("people_per_house",ratio_pipeline(),["population","households"]),
#     ("log",log_pipeline, ["total_bedrooms", "total_rooms", "population",
#                                "households", "median_income"]),
#     ("geo",cluster_simil,['latitude','longitude']),
#     ("cat",cat_pipeline,make_column_selector(dtype_include=object)) ],
                                 
#      remainder = default_num_pipeline)
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  
housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)
preprocessing.get_feature_names_out()
print(pd.DataFrame(housing_prepared,columns = preprocessing.get_feature_names_out()))


# %% [markdown]
# # ***SELECTING AND TRAINING A MODEL***

# %% [markdown]
# *Train and Evaluate on the Training Set :*

# %%
from sklearn.linear_model import LinearRegression
lin_reg = make_pipeline(preprocessing,LinearRegression())
lin_reg.fit(housing,housing_labels)

# %%
housing

# %%
housing_predictions = lin_reg.predict(housing)
housing_predictions[:7].round(-2)

# %%
housing_labels.iloc[:5].values

# %%
# as we chose rmse as performance measure hence evaluationg :
from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(housing_labels, housing_predictions)
print(np.sqrt(lin_rmse))
  

# %% [markdown]
# The error of $69k is not impressive hence we can say that the model is underfitting the trtaining data. Which means the features do not provide enough information or the model is not powerful to make good predictions.
# 
# The main ways to fix hte underfitting problem is by :
# - selecting a powerful model.
# - feed the algo better features
# - or reduce the constraints (but here the model is not regualrized hence this option is out.)
#   

# %%
# trying decision tree regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing,DecisionTreeRegressor(random_state =42))
tree_reg.fit(housing,housing_labels)

# %%
housing_predictions = tree_reg.predict(housing)
tree_rmse=mean_squared_error(housing_labels,housing_predictions)
print(np.sqrt(tree_rmse))

# %% [markdown]
# The zero error shows that the model has badly overfit the data. As we we sw earlier we dint want to touch the test set until our model is ready to launch. So we need the part of training set for training and part of it for model validation.

# %% [markdown]
# *Better evaluation using cross-validation :*

# %% [markdown]
# - one way to evaluate is by splitting the data using train test split and the splittin the data into a small training set and a validation set for evaluation.
# - A great alternative is using sklearns k-fold crosss validation.
# - How cross validation wroks ?
#    - The training set is generally split into 10 random and non overlapping sub-sets called folds.
#    - The model is trained and evaluated 10 times, 9 folds are kept for training the model and the
#      remaining fold is used to evaluate the model.
#    - This process is repeated 10 times each time using a different fold for validation.
#    - Finally we get an array of 10 evaluation scores.

# %%
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg,housing,housing_labels,
                            scoring= "neg_root_mean_squared_error",cv =10)

# code explanation :
# - the scoring parameter here tells the sklearn how to evaluate the models and the sklearn expects the greater is better
# - the sklearn automatically here expects a utility function (greater is better).
# - but here we are dealing with errors which should be lower is better , hence we use scoring method 
#   to be neg_mse , which will flip and the largest error will be smallest which means it is not a good thing.
# - but the draw back is the negative values of the errors are returned so to get the real positive values
#   we add a minus sign.

pd.Series(tree_rmses).describe()

# %% [markdown]
# Now we can see that the decision tree doesnt perform well and performs almost as bad as the linear regression. Hence the cross validation not only tells about the performance but also tells how much precise the model is. But it requires training the model several times which is not always feasible.
# 
# Now Lets try another model : RandomForestRegressor.
# 
# -  random forests work by training many decision trees on random
#  subsets of the features, then averaging out their predictions. Such models
#  composed of many other models are called ensembles: they are capable of
#  boosting the performance of the underlying model.
# 

# %%
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
from sklearn.model_selection import cross_val_score

# Train once and store
forest_rmses = -cross_val_score(
    forest_reg,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)


# %%
pd.Series(forest_rmses).describe()

# %%
preprocessing

# %% [markdown]
# Seeing above stats the random forest looks quite promising. However if wee train the random forest on the training set and calculate rmse it comes out to be 17k : thats muxh lower lower  hence alot of overfitting going on still. Therefore before diving into random forest wee must select more models to work on. Without spending too much time tweaking the hyperparamters. The goal is to shortlist a few two to five promising models.

# %% [markdown]
# # ***FINE TUNE YOUR MODEL***

# %% [markdown]
#   Assuming we have shortlisted few models, we now need to fine tune them.

# %%
# making pipeline

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
 ])
full_pipeline

# %% [markdown]
# ***Grid Search***
#  - Instead doing the hyperparameter finding manually we can do it using  GridSearchCV class.
#  - how to use ?
#     - provide the full pipeline first
#     - then in the param_grid we define our search space.
#     - param_grid is a list of dictionaries where each dictionary represents a grid of paramteres
#       to search.
#     - Parameters are written using the syntax: "pipelineStepName__substep__param_name"
#       for example : { "preprocessing__geo__n_clusters" : [5,8,10] in list we provide the values we
#       want to try.
#       Another example : "random_forest__max_features" :[4,6,8] } , similarly we can provide
#       another dictionary of the parameters we want to tune.

# %%
# performing grid search :
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]}, # we are fine tuning using max features because it tell how better the model random forest will generalize as it tells how many features to consider at each fold
                                                # the grid search finds out considering how many features is best ( considering lower featrures help in better genralizayion and prevents overfitting)
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3, # tree fold cross validation
    scoring= 'neg_root_mean_squared_error'
) 

grid_search.fit(housing,housing_labels)
grid_search.best_params_



# %%
grid_search.best_estimator_

# %% [markdown]
# 🔁 GridSearchCV expands it like this:
# Each dictionary is handled separately as a grid of combinations:
# 
# First dictionary:
# 
# n_clusters: [5, 8, 10] → 3 values
# 
# max_features: [4, 6, 8] → 3 values
# ⏩ 3 × 3 = 9 combinations
# 
# Second dictionary:
# 
# n_clusters: [10, 15] → 2 values
# 
# max_features: [6, 8, 10] → 3 values
# ⏩ 2 × 3 = 6 combinations
# 
# 🔄 Total combinations = 9 + 6 = 15
# 
# ➗ Cross-Validation:
#  cv= 3
# 
#  total model trainings = 15x3 =45.
#  
# 

# %% [markdown]
# If grid search cv is initialised with refit = True (which is default), then once after finding the est estimatoors using cross validation, it retrains it on the whole training set (i.e after training and evaluating on the folds using the cross validation method the gridsearch retrains the model on the complete training set again after it has found the best estimators)

# %%
# Getting the evaluation scores of the modals after different hyperparamter tunings using
# grid_search.cv_results_  which returns dictionary which we can convert in a dataframe:
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score",ascending = False , inplace =True)
cv_res

# %%
# making dataframe nicer : 
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# extra code – these few lines of code just make the DataFrame look nicer
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

cv_res.head()

# %% [markdown]
# The mean test rmse score for the best model is 43832 as we can see above is quite better than the earlier which was 48k (before the changing the hyperparameter values).HENCE WE HAVE FINE TUNED OUR MODEL USING GridSearchCV and cross validation.
# 

# %% [markdown]
# ***Random Search***
# 
# The randomised search is useful when :
# - you have many hyperparameter
# - or some of them are continuous
# - or youre short on computing power
# 
# Whats RandomizedSearchCV ? 
# - randomly samples a fixed number of hyperparameter combinations for a search space we proovide.
# - uses cross validation for evaluation.
# - allows us to controol the number fo iteration using parameter n_iter.
#   
# 

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3,high =50), #any integer form 3 to 50',
    "random_forest__max_features": randint(low=2,high=20)
}

rnd_search = RandomizedSearchCV(
    full_pipeline,param_distributions= param_distribs,n_iter=10,cv=3
    ,scoring = 'neg_root_mean_squared_log_error',random_state=42
)
rnd_search.fit(housing,housing_labels)

# %%
cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()

# %% [markdown]
# ***HalvingRandomSearchCV and HalvingGridSearchCV***
# 
#  are resource-efficient hyperparameter search methods — part of what’s called successive halving algorithms. Instead of training all candidates on full resources they train on limited resources and then check the best survivors , then the survivors get mroe training data and this is repeated until few best are found.
#  

# %% [markdown]
# Step-by-Step: How Successive Halving Works :
# 1. Generate many candidates, from a grib or random distribution.
# 2. Round 1 : Train on few resources , small data portion, evaluate cv, keep top X%.
# 3. round 2 : Train survivors on ore data, keep top X%.
# 4. Repeat until final round : final few models trained on full data , best model wins.

# %%
# from sklearn.experimental import enable_halving_search_cv  # Needed to enable!
# from sklearn.model_selection import HalvingGridSearchCV

# halving_search = HalvingGridSearchCV(
#     estimator=full_pipeline,
#     param_grid=param_grid,  # or param_distributions if Random
#     factor=3,               # how aggressively to reduce candidates (default=3)
#     scoring="neg_root_mean_squared_error",
#     cv=3
# )
# halving_search.fit(housing, housing_labels)

# %% [markdown]
# ***Ensemble method***
# 
# Another method is using ensembel / combining methods which refers to the combining of the bes tmodels predictions with other models.
# As even the best model found can makke certain types of errors hence its better to combine the twoo different type of models which produces two different types of errors for better predictions, this prevents overfitting and produces better results. The errors of the two models are not correlated they make errors in two different areas. For example combining the kmeans and random forest.

# %% [markdown]
# # ***ANALYZING THE BEST MODELS AND THEIR ERRORS:***
# 

# %%
final_model = rnd_search.best_estimator_
feature_importance = final_model["random_forest"].feature_importances_
print(pd.DataFrame(feature_importance))

# %% [markdown]
# The above code returns an array which contains the importance of all the features in predicting the accurate values of the target value.
# Seeing the above returned array it is evident that there are alot of useless or unimportant features in the dataset which we want to remove so that the model doesnt get confused andwe can get better results.
# Our model produces certain types of errors and we should try to minimize them somehow such as by dropping the uninformative features, by adding new features to the datasetor getting rid of outliers.
# It should be notes that our model should not just perform better only on the average but also on the all districts of other regions as well it shouldnt matter if the dist are in norhtern or southern or in a rural or urban area. If the model doesnt perform well on these areas then the model shouldnt be deployed.

# %%
full_pipeline[0].get_feature_names_out()

# %%
len(full_pipeline[0].get_feature_names_out())

# %%
sorted(zip(
    feature_importance,final_model["preprocessing"].get_feature_names_out()),reverse = True
)

# %% [markdown]
# How 45 clusters were formed only for the geo feature why not for the max feature ? 
# - after alot of cv the model finds out there should be 45 clusters in the given data for better tuning (using rnd_search)
# - hence the 45 clusters were created as the parameter was n_clusters and then the similarity of each cluster center is measured with each districts
#   location on the map.
# - and the clusters were not created for the max features part cause the purpose was not create the features it was just tried internally at the
#   backend that how the model performs on tweaking the parameters.

# %% [markdown]
# # ***Evaluate Your System on the Test Set***

# %%
X_test = strat_test_set.drop("median_house_value",axis = 1)
y_test = strat_test_set["median_house_value"].copy()

# %%
final_predictions = final_model.predict(X_test)

# %%
final_predictions

# %%
housing_labels

# %%
final_rmse = mean_squared_error(final_predictions,y_test)

# %%
np.sqrt(final_rmse)

# %%
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors)))

# %% [markdown]
# Here we are using the confidence interval because we cant say that the error that will occur will be essentially eqaul to 39k so to get a better estimate and range we find confidence interval.
# 

# %% [markdown]
# # ***Launch, Monitor, and Maintain Your System***

# %% [markdown]
# - As we are done with almost everything now its time to deploy the model.
# - we deploy the model on joblib and load it whenever need on any system.
# - But to use it again we will have to import or re write the custom transformers we coded during the model training.

# %%
 import joblib
 joblib.dump(final_model, "my_california_housing_model.pkl")

# %%



