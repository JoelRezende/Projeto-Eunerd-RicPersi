import numpy as np
import pandas as pd
#import tensorflow as tf
#import plotly.graph_objects as go
import math
from sklearn.preprocessing import OneHotEncoder
#from tensorflow import keras
from matplotlib import pyplot as plt
#from plotly.subplots import make_subplots

print ("hello world1")

# %matplotlib inline
def main():
    data = get_data()
    data = clean_data(data)
    dashboards(data)
    data = dummies(data)

    return data

def get_data(): 
    csv_name = "Experiencia2.csv"
    data = pd.read_csv(csv_name, sep=";", encoding= 'utf-8')

    return data


def clean_data(data):

    contagem_valores_time = data.Time.value_counts()

    data['Distance_m'] = data[pd.to_numeric(data['Distance(m)'], errors='coerce').notnull()]['Distance(m)'].astype(float)
    data['Distance_m'] = data[pd.to_numeric(data['Distance(m)'], errors='coerce').notnull()]['Distance(m)'].astype(float)
    data['Distance_km'] = data.Distance_m /1000
    data = data[data.Distance_km.between(data.Distance_km.quantile(.01), data.Distance_km.quantile(.99))]
#data.hist(column='Time', bins=50)
    data = data.drop([  ], axis=1)
    data['Hour_of_day'] = [x[12:14] for x in data['Time']]
    data = data[pd.to_numeric(data.Hour_of_day, errors='coerce').notnull()]
    data = data.astype({'Hour_of_day': int})
#data.Hour_of_day.value_counts()
    data = data.drop(['Type', 'Team_Name', 'Notes', 'Agent_ID', 'Agent_Name', 'Customer_Name', 'CLIENTE', 'Pricing', 'Task_Type', 'Task_Status', 'Plano', 'Total_Horas', 'Valor_Hora_Adicional_R', 'Earning'], axis=1)

    data = data.drop_duplicates(subset='Task_ID', keep="last", )
    data = data.reset_index(drop=True)

    data.Segmento = data.Segmento.replace(['hardware'], 'Hardware')
    data.Segmento = data.Segmento.replace(['TI'], 'Ti')
    data = data.drop(['Time', 'Task_ID'], axis=1)
    # data.Segmento.value_counts()

    data = data[pd.to_numeric(data['Distance(m)'], errors='coerce').notnull()]
    data['Distance(m)'] = data['Distance(m)'].astype(float)
    data = data[pd.to_numeric(data['Total_Time_Taken(min)'], errors='coerce').notnull()]
    data['Total_Time_Taken(min)'] = data['Total_Time_Taken(min)'].astype(float)
    data = data[pd.to_numeric(data.Valor_R, errors='coerce').notnull()]
    data.Valor_R = data.Valor_R.astype(float)

    data = data[data.Valor_R.between(data.Valor_R.quantile(.05), data.Valor_R.quantile(.95))]
    
    data = data[data['Distance(m)'].between(data['Distance(m)'].quantile(.01), data['Distance(m)'].quantile(.99))]

    data['Distance(m)'] = data['Distance(m)']/1000
    data.rename(columns={'Distance(m)':'Distance(km)'}, inplace=True)
  

    data = data.drop(['Total_Time_Taken(min)'], axis=1)
    return data


def dashboards(data):
    fig = data.Total_Horas.value_counts().plot.bar()
    fig.savefig("Horas.jpg")

    fig = cv2.imread('Horas.jpg')
    # codificar imagem como jpeg  
    
    data_valor_r = data.Valor_R
    prob = data_valor_r.value_counts(normalize=True, bins=5)
    threshold = 0.01
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    #prob['other'] = tail_prob
    Valor_R_bar = prob.plot(kind='bar')   
    Valor_R_bar.save("Valor_R.png") 
    # data.hist(column='Valor_R', bins=100, grid=False)

    total_items = len(data.columns)
    items_per_row = 2
    total_rows = math.ceil(total_items / items_per_row)
    fig = make_subplots(rows=total_rows, cols=items_per_row)
    cur_row = 1
    cur_col = 1
    for index, column in enumerate(data.columns):
        fig.add_trace(go.Box(y=data[column], name=column), row=cur_row, col=cur_col)
        
        if cur_col % items_per_row == 0:
            cur_col = 1
            cur_row = cur_row + 1
        else:
            cur_col = cur_col + 1
        
    fig.update_layout(height=1000, width=550,  showlegend=False)
    fig.save("boxplot.png")
    
    histograma_tot = data.hist(column='Distance(km)', bins=50)
    histograma_top_50 = data[data['Distance(km)']<50].hist(column='Distance(km)', bins=50)
    histograma_top_5 = data[data['Distance(km)']<5].hist(column='Distance(km)', bins=50)
    #data['Distance(km)'][data['Distance(km)']].count()
    histograma_tot.save("hist_tot.png")
    histograma_top_50.save("hist_top50.png")
    histograma_top_5.save("hist_top_5.png")

    return horas_encoded


 
    
    #fig.show()

#data.plot.scatter(x='Distance(km)', y='Valor_R')

#data.Tamanho.value_counts()

#data = data[data['Total_Time_Taken(min)'].between(data.Valor_R.quantile(.05), data.Valor_R.quantile(.95))]

#data.Hour_of_day.value_counts().sort_values().plot(kind = 'barh')

def dummies(data):
    tamanho_dummies = pd.get_dummies(data.Tamanho, prefix='Tamanho').iloc[:, 1:]
    segmento_dummies = pd.get_dummies(data.Segmento, prefix='Segmento').iloc[:, 1:]
    data = data.drop(['Tamanho', 'Segmento'], axis=1)
    data = pd.concat([data, tamanho_dummies, segmento_dummies], axis=1)
    return data


print(main())