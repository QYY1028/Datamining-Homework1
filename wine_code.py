import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import datasets,ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

data_name = r'D:\研一下学期\数据挖掘\互评作业1\Wine reviews\winemag-data_first150k.csv'
data_name2 = r'D:\研一下学期\数据挖掘\互评作业1\Wine reviews\winemag-data-130k-v2.csv'
txt=open(r'D:\研一下学期\数据挖掘\互评作业1\Wine_reviews_result1.txt','w',encoding = 'utf-8')
problem_1 = 0
def fiveNumber(nums):
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.percentile(nums,25)
    Median=np.median(nums)
    Q3=np.percentile(nums,75)
    
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR #下限值
    upper_limit=Q3+1.5*IQR #上限值
    
    return Minimum,Q1,Median,Q3,Maximum,lower_limit,upper_limit

def set_missing_ages(df):
    age_df = df[['price','points']]

    known_age = age_df.loc[(age_df['price'].notnull())]
    unknown_age = age_df[(age_df['price'].isnull())]

    y = known_age.values[:, 0]
    X = known_age.values[:, 1:]
    
    # fit到RandomForestRegressor之中
    rfr = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age.values[:, 1:])

    df.loc[ (df['price'].isnull()), ['price']] = predictedAges 

    return df, rfr

def knn_missing_filled(data, k = 3, dispersed = True):

    data_lost4 = data.copy()
    x_train = data_lost4[data_lost4.price.notnull()]['points'].values.reshape(-1,1)
    y_train = data_lost4[data_lost4.price.notnull()]['price'].values.reshape(-1,1)
    # print(len(x_train))
    # print(len(y_train))
    test = data_lost4[data_lost4.price.isnull()]['points'].values.reshape(-1,1)

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    data_lost4.loc[ (data_lost4.price.isnull()), 'price' ] = clf.predict(test)

    return data_lost4

file = pd.read_csv(data_name)
# print(file.type)
print(file.columns)
atts = file.columns

print(file['price'].isnull())
print(type(file['price'].isnull()))
num=0
if problem_1:
  for i in atts[1:]:
    print(i)
    if i!= 'points' and i!= 'price' and i!='description':
        dict = {}
        for j in file.loc[:,i]:
            if j in dict.keys():
                dict[j]+=1
            else:
                dict[j]=1
                num+=1
    
            
        txt.write(i+'\n')
        txt.write(str(dict))
        txt.write('\n')
        print(num)
  txt.close()

null_test = file['region_2'][1]
print(math.isnan(null_test))
# data_c,data_cc = set_missing_ages(file)
# data_c.to_csv(r'D:\研一下学期\数据挖掘\互评作业1\new_wine.csv')
# print(data_c.type)

att_pool = ['points', 'price']
for att in att_pool:
  num_list = []
  # dict_hist = {}
  nan_count = 0
  for point in file.loc[:,att]:
   if not math.isnan(point):
    num_list.append(point)
   else:
    nan_count+=1
  if att == 'points':
    bins_set = 10
  else:
    bins_set=1000
  point_np = np.array(num_list)
  minn,Q1,mediann,Q3,maxx,lower_limit,upper_limit = fiveNumber(num_list)
  print(att+' five number:')
  print(minn,Q1,mediann,Q3,maxx,lower_limit,upper_limit)
  print('nan count:'+str(nan_count))
'''
'''
#hist
  plt.hist(num_list,edgecolor='k', alpha=0.35,bins=bins_set)
  plt.show()
  # 
#boxplot
  fig = plt.figure(figsize=(8, 6))
  plt.boxplot(num_list, notch=False, sym='o', vert=True)

  # plt.xticks([x+1 for x in range(len(num_list))], ['x1', 'x2', 'x3', 'x4'])
  plt.title('box plot:'+att)
  plt.xlabel('x')
  plt.show()

#for data lose
#b
  ser = pd.Series(point_np)
  mode_data = ser.mode()
  fix_list = []
  for point in file.loc[:,att]:
   if not math.isnan(point):
    fix_list.append(point)
   else:
    fix_list.append(mode_data)
  fig = plt.figure(figsize=(8, 6))
  plt.boxplot(num_list, notch=False, sym='o', vert=True)
  plt.title('box plot:'+att)
  plt.xlabel('x')
  plt.show()
  # 
#c
data_c,data_cc = set_missing_ages(file)
data_c.to_csv(r'D:\研一下学期\数据挖掘\互评作业1\new_wine.csv')
att='price'
fix_list = []
nan_count = 0
for point in data_c.loc[:,att]:
 if not math.isnan(point):
  fix_list.append(point)
 else:
  nan_count+=1
fig = plt.figure(figsize=(8, 6))
plt.boxplot(fix_list, notch=False, sym='o', vert=True)
plt.title('box plot:'+att)
plt.xlabel('x')
plt.show()
'''
'''
#d
data_d = knn_missing_filled(file)
att='price'
fix_list = []
nan_count = 0
for point in data_d.loc[:,att]:
 if not math.isnan(point):
  fix_list.append(point)
 else:
  nan_count+=1
fig = plt.figure(figsize=(8, 6))
plt.boxplot(data_d['price'], notch=False, sym='o', vert=True)
plt.title('box plot:'+att)
plt.xlabel('x')
plt.show()
'''
