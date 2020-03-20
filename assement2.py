import random
import pandas as pd
import numpy as np

def create_sample_data():
  '''
    @author: ankush mishrikotkar
    @description: this functions generates the sample data that includes one binary labeled target feature, two numeric columns that contain random generated numbers and three categorical features having less or more unique values
    @input_params: None
    @output_params: df
  '''
  data = {
  "target": np.random.randint(0,2,100),
  "col1": np.random.random(100),
  "col2": np.random.random(100),
  "col3": [random.choice(
    ['hot','cold','warm','humid','freez','sunny',
    'foggy','raining','snow']) for i in range(100)],
  "col4": [random.choice(
    ['pune','mumbai','delhi','bangalore',
    'kolkata','nagpur','hyderabad','chennai','indore']) for i in range(100)],
  "col5": [random.choice(
    ['type_a', 'type_b', 'type_c']) for i in range(100)]
  }

  df = pd.DataFrame(data)
  print("\nSample Data:")
  print(df.head(10))
  print("\nColumns:",list(df.columns))
  return df

def encode_categorical_column(input_data, column_number):
  '''
    @author: ankush mishrikotkar
    @description: this function checks multiple conditions to encode the categorical feature column into the numeric.
    1. if the selected column is target, the fuction will not return anything
    2. if the selected column is numeric, the function will return the original dataset
    3. if the selected column is categorical and it contains less than or equal to 5 unique values, the one hot encoded from the scikit-learn strategy will be applied
    4. if the selected column is categorical and it contains more than 5 unique values, the mean encoding strategy will be applied
    > How the mean encoding works:
    1. Compute global mean of the target labels
    2. Group by the selected categorical variable and obtain the aggregated count over the target variable
    3. Group by the selected categorical variable and obtain the aggregated mean over the target variable
    4. To encode smooth encoding intialise random weights
    5. Compute the smoothed encoding using the computation
    6. Computation formula:
    smoothed_encoding = (groupby_calculated_count*groupby_calculated_sum+random_weights*global_mean)/(groupby_calculated_count+random_weights)
    @input_params: input_data, column_number
    @output_params: df_new
  '''
  if column_number > 1 and column_number <= len(input_data.columns):
    if input_data[list(input_data.columns)[column_number-1]].dtypes != object:
      print("\nSelected column is Numeric.")
      return input_data
    else:
      if len(list(input_data[list(input_data.columns)[column_number-1]].unique())) <= 5:
        from sklearn.preprocessing import OneHotEncoder
        ohc = OneHotEncoder()
        column_name = list(input_data.columns)[column_number-1]
        #print("\nColumn Name:",column_name)
        #print("\nNumber of unique values in Column: {}".format(column_name))
        #print("Unique Values:",list(input_data[column_name].unique()))
        #print("Unique Count:",len(list(input_data[column_name].unique())))
        ohe = ohc.fit_transform(input_data[column_name].values.reshape(-1,1)).toarray()
        dfOneHot = pd.DataFrame(ohe, 
          columns = [column_name + "_" + str(ohc.categories_[0][i])
           for i in range(len(ohc.categories_[0]))]
          )
        df_new = pd.concat([input_data, dfOneHot], axis = 1)
        df_new.drop(columns=[column_name],axis=1,inplace=True)
        #print("\nTransformed Data:")
        #print(df_new.head())
        return df_new
    
      else:
        column_name = list(input_data.columns)[column_number-1]
        #print("\nColumn Name:",column_name)
        #print("\nNumber of unique values in Column: {}".format(column_name))
        #print("Unique Values:",list(input_data[column_name].unique()))
        #print("Unique Count:",len(list(input_data[column_name].unique())))
        # Compute the global mean
        mean_target = input_data['target'].mean()
        # Compute the number of values and mean of each group
        aggregate = input_data.groupby(column_name)['target'].agg(['count','mean'])
        counts = aggregate['count']
        means = aggregate['mean']
        weight = 100
        # Compute the Smoothed mean
        smooth_mean = (counts*means+weight*mean_target)/(counts+weight)
        # Replace each value by the according smoothed mean
        df_new = input_data.copy()
        df_new.loc[:,column_name+"_smooth_encoding"] = input_data[column_name].map(smooth_mean)
        df_new.drop(columns=[column_name],axis=1,inplace=True)
        #print("\nTransformed Data:")
        #print(df_new.head())
        return df_new
  else:
    print("\nSelected column is the Target Column")


df = create_sample_data()
df_new = encode_categorical_column(df,column_number=5)
print("\nTransformed Data:")
print(df_new.head(10))
print("\nColumns:",list(df_new.columns))
