#!/usr/bin/env python
# coding: utf-8

# In[25]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


# In[10]:


breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.feature_names)
print(breast_cancer_data.data[0])


# In[12]:


print(breast_cancer_data.target)
print(breast_cancer_data.target_names)


# In[39]:


training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=1)


# In[40]:


print(len(training_data))
print(len(training_labels))


# In[41]:


accuracy = []
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracy.append(classifier.score(validation_data, validation_labels))


# In[42]:


classifier.fit(training_data, training_labels)


# In[43]:


print(classifier.score(validation_data, validation_labels))


# In[44]:


k_list = range(1, 100)
plt.plot(k_list, accuracy)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()


# In[ ]:




