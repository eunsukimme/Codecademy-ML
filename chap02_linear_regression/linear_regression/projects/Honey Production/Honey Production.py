#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# In[4]:


df = pd.read_csv('honeyproduction.csv')
print(df.head())


# In[17]:


# 각 년도별 totalprod의 평균을 계산합니다
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)


# In[47]:


# 위 데이터프레임에서 year를 X변수에 저장합니다
X = prod_per_year['year']
# (15, ) 꼴의 데이터를 (15, 1)의 행렬로 변환시킨다
X = X.values.reshape(-1, 1)
print(X)
print(X.shape)


# In[49]:


# 위 데이터프레임에서 totalprod를 y변수에 저장합니다
y = prod_per_year['totalprod']
y = y.values.reshape(-1, 1)
print(y)
print(y.shape)


# In[50]:


plt.plot(X, y, 'o')
plt.show()


# In[55]:


# 선형 회귀 모델을 생성하고 fit 해봅시다
regr = LinearRegression()
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)


# In[57]:


# 모델이 predict 한 값을 함께 그래프로 그려봅시다
y_predict = regr.predict(X)
plt.plot(X, y, 'o')
plt.plot(X, y_predict)
plt.show()


# In[65]:


# 앞으로의 추이를 예측해보기 위해 연도 데이터를 더 생성해 봅시다
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)
print(X_future[:10])


# In[67]:


# 이제 앞으로의 연도별 totalprod를 예측해 봅시다
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict)
plt.show()


# In[ ]:




