# -*- coding: utf-8 -*-
"""Experiencia.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DyiQzfxiygsktCFqDWiECQU0TpoTGiP-
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from matplotlib import pyplot as plt

# %matplotlib inline
def main():
    data = get_data()
    data = clean_data(data)
    


def get_data(): 
    csv_name = "Experiência2.csv"
    data = pd.read_csv(csv_name, sep=";", encoding= 'utf-8')

    return data


def clean_data(data):

    contagem_valores_time = data.Time.value_counts()

    data['Distance_m'] = data[pd.to_numeric(data['Distance(m)'], errors='coerce').notnull()]['Distance(m)'].astype(float)
    data['Distance_m'] = data[pd.to_numeric(data['Distance(m)'], errors='coerce').notnull()]['Distance(m)'].astype(float)
    data['Distance_km'] = data.Distance_m /1000
    data = data[data.Distance_km.between(data.Distance_km.quantile(.01), data.Distance_km.quantile(.99))]
#data.hist(column='Time', bins=50)
    return data

def dashboards(data):
    fig = data.Total_Horas.value_counts().plot.bar()
    fig.savefig("Horas.jpg")

    fig = cv2.imread('Horas.jpg')
    # codificar imagem como jpeg
    horas_encoded = cv2.imencode('Horas.jpg', fig)
    
    return horas_encoded


df = data.copy(deep=True)
df = df.drop([  ], axis=1)

df['Hour_of_day'] = [x[12:14] for x in df['Time']]
df = df[pd.to_numeric(df.Hour_of_day, errors='coerce').notnull()]
df = df.astype({'Hour_of_day': int})
df.Hour_of_day.value_counts()

df = df.drop(['Type', 'Team_Name', 'Notes', 'Agent_ID', 'Agent_Name', 'Customer_Name', 'CLIENTE', 'Pricing', 'Task_Type', 'Task_Status', 'Plano', 'Total_Horas', 'Valor_Hora_Adicional_R', 'Earning'], axis=1)

df = df.drop_duplicates(subset='Task_ID', keep="last", )
df = df.reset_index(drop=True)

df.Segmento = df.Segmento.replace(['hardware'], 'Hardware')
df.Segmento = df.Segmento.replace(['TI'], 'Ti')
df = df.drop(['Time', 'Task_ID'], axis=1)
df.Segmento.value_counts()

df = df[pd.to_numeric(df['Distance(m)'], errors='coerce').notnull()]
df['Distance(m)'] = df['Distance(m)'].astype(float)
df = df[pd.to_numeric(df['Total_Time_Taken(min)'], errors='coerce').notnull()]
df['Total_Time_Taken(min)'] = df['Total_Time_Taken(min)'].astype(float)
df = df[pd.to_numeric(df.Valor_R, errors='coerce').notnull()]
df.Valor_R = df.Valor_R.astype(float)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
total_items = len(df.columns)
items_per_row = 2
total_rows = math.ceil(total_items / items_per_row)
fig = make_subplots(rows=total_rows, cols=items_per_row)
cur_row = 1
cur_col = 1
for index, column in enumerate(df.columns):
    fig.add_trace(go.Box(y=df[column], name=column), row=cur_row, col=cur_col)
    
    if cur_col % items_per_row == 0:
        cur_col = 1
        cur_row = cur_row + 1
    else:
        cur_col = cur_col + 1
    
fig.update_layout(height=1000, width=550,  showlegend=False)
fig.show()

df = df[df.Valor_R.between(df.Valor_R.quantile(.05), df.Valor_R.quantile(.95))]

df = df[df['Distance(m)'].between(df['Distance(m)'].quantile(.01), df['Distance(m)'].quantile(.99))]

df['Distance(m)'] = df['Distance(m)']/1000
df.rename(columns={'Distance(m)':'Distance(km)'}, inplace=True)
df

df = df.drop(['Total_Time_Taken(min)'], axis=1)

s2 = df.Valor_R
prob = s2.value_counts(normalize=True, bins=5)
threshold = 0.01
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
#prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=25)
plt.show()

df.hist(column='Valor_R', bins=100, grid=False)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
total_items = len(df.columns)
items_per_row = 2
total_rows = math.ceil(total_items / items_per_row)
fig = make_subplots(rows=total_rows, cols=items_per_row)
cur_row = 1
cur_col = 1
for index, column in enumerate(df.columns):
    fig.add_trace(go.Box(y=df[column], name=column), row=cur_row, col=cur_col)
    
    if cur_col % items_per_row == 0:
        cur_col = 1
        cur_row = cur_row + 1
    else:
        cur_col = cur_col + 1
    
fig.update_layout(height=1000, width=550,  showlegend=False)
fig.show()

df.plot.scatter(x='Distance(km)', y='Valor_R')

df.Tamanho.value_counts()

#df = df[df['Total_Time_Taken(min)'].between(df.Valor_R.quantile(.05), df.Valor_R.quantile(.95))]

df.Hour_of_day.value_counts().sort_values().plot(kind = 'barh')

df

tamanho_dummies = pd.get_dummies(df.Tamanho, prefix='Tamanho').iloc[:, 1:]
segmento_dummies = pd.get_dummies(df.Segmento, prefix='Segmento').iloc[:, 1:]
df = df.drop(['Tamanho', 'Segmento'], axis=1)
df = pd.concat([df, tamanho_dummies, segmento_dummies], axis=1)
df

df.hist(column='Distance(km)', bins=50)
df[df['Distance(km)']<50].hist(column='Distance(km)', bins=50)
df[df['Distance(km)']<5].hist(column='Distance(km)', bins=50)
#df['Distance(km)'][df['Distance(km)']].count()
plt.show()

#df.drop(columns=['Distance(km)'], axis=1)
df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="Valor_R").values, df.Valor_R,test_size=0.2) #, random_state=123)

print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)
print(X_test.shape, X_test.dtype)
print(y_test.shape, y_test.dtype)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', name='dense_1'))
model.add(Dropout(rate=0.3, name="drop_1"))
model.add(Dense(64, activation='relu', name='dense_2'))
model.add(Dense(1, activation='linear', name='dense_output'))
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

list_callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True),
    ModelCheckpoint(filepath="/content/best_model.h5", monitor="val_loss", verbose=1, save_best_only=True),
]

hist = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=400,
    verbose=1,
    callbacks=list_callbacks,
    validation_data=(X_test, y_test)
)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import numpy as np

fig = go.Figure()
fig.add_trace(go.Scattergl(y=hist.history['loss'],
                    name='Train'))
fig.add_trace(go.Scattergl(y=hist.history['val_loss'],
                    name='Valid'))
fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='Loss')
fig.show()

random_idx = np.random.choice(df.shape[0])
random_row = df.iloc[random_idx].to_frame().T
random_row.head()

_, df_test = df.align(random_row, join="left", axis=1, fill_value=0)
df_test

x = df_test.drop(columns=["Valor_R"]).values
y = df_test['Valor_R'].values

print("   real:", y)
print("predict:", model.predict(x))

df.to_csv(r'/content/csv/Experiência.csv', index = False)

input_dict = {
    "value": 50,
    "plan_type": "ALCANCE",
    "city_size_client": "small",
    "categoria": "InfraestruturadeRedesePeriféricos",
    "serviço_2": "Crimpagemdecabos",
    "status": "Concluído"
}