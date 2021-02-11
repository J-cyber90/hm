#!/usr/bin/env python
# coding: utf-8

# In[4]:


#การป้อนคำสั่งเลือกฟังค์ชั่น เพื่อให้ระบบอ่านข้อมูลได้
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[5]:


#การให้ระบบแจ้งเวอร์ชั่นของฟังค์ชั่นที่จะใช้ เพื่อประเมินว่าโค้ดที่คัดลอกมาจะใช้กับเวอร์ชั่นนี้ได้หรือไม่
print(f'pandas  version = {pd.__version__}')
print(f'numpy   version = {np.__version__}')
print(f'seaborn version = {sns.__version__}')


# In[6]:


#การเช็คระบบเวลา เพื่อดูอาการดีเลย์ของระบบ
pd.Timestamp.now()


# In[7]:


#การนำเอาลิ้งค์จากไฟล์โจทย์ที่อาจารย์แชร์ใน github มาลงให้ระบบอ่านข้อมูล และสุ่มตัวอย่างมา 10 ตัว  
df=pd.read_excel('https://github.com/thagoon0931385877/E-logistic/raw/main/test.xlsx')
df.sample(10)


# In[8]:


#การบ่งบอกถึงหัวข้อแถวของตาราง
df.columns


# In[10]:


#การสั่งให้ระบบนำเอาข้อมูลทั้ง 200 ตัว มาทำเป็นกราฟแบบกระจายตัว
sns.scatterplot(data=df, x='X', y='Y')


# In[11]:


#การสุ่มเลขจากตัว x มา 3 ตัว
rx=np.random.uniform(1, 7, 3)
rx


# In[12]:


#การสุ่มเลขจากตัว y มา 3 ตัว
ry=np.random.uniform(0, 2.5, 3)
ry


# In[13]:


#การนำเอาตัวสุ่มทั้ง 6 มาเชื่อมกัน ก็ได้จุด diamon สีแซลม่อนมา 3 ตัว
sns.scatterplot(data=df, x='X', y='Y')

plt.scatter(rx, ry, color='salmon', marker='D')


# In[14]:


from sklearn.cluster import KMeans


# In[15]:


#การสั่งระบบให้สร้างรูปแบบข้อมูลออกเป็น 3 กลุ่ม
model=KMeans(n_clusters=3)
model


# In[17]:


#การกำหนดให้ระบบแบ่งจุด xy ให้เป็น 3 กลุ่ม
X=df[['X', 'Y']]
model.fit(X)


# In[18]:


#การสั่งให้ระบบหาตำแหน่งจุดที่อยู่ศูนย์กลางของแต่ละกลุ่ม
model.cluster_centers_ # centroids


# In[20]:


#การพอทจุดระบุตำแหน่งของจุดที่อยู่เป็นศูนย์กลางของแต่ละกลุ่ม
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1])


# In[21]:


#การระบุและจำแนกว่าจุดทั้งหมด จำนวน 200 ตัว อยู่ในกลุ่มไหนบ้าง ซึ่งมีทั้งหมด 3 กลุ่ม ได้แก่ 0 1 และ 2
model.labels_


# In[22]:


#การแสดงผลลัพธ์ว่าแต่ละจุดอยู่ในกลุ่มใดบ้าง
df['cluster']=model.labels_
df


# In[25]:


#การแสดงกราฟแบบกระจายข้อมูล เพื่อให้เห็นว่าแต่ละกลุ่มมีจุดใดบ้าง และมีจุดศุนย์กลางอยู่ที่ใด ซึ่งสามารถแบ่งได้ตามหลักการแบ่งแยกสีกลุ่ม
sns.scatterplot(data=df, x='X', y='Y', hue='cluster', alpha=.5, palette=['green', 'blue', 'orange'])
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color='.1', marker='D')

