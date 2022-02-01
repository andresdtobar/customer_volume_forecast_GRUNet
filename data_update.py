import pandas as pd
import numpy as np
import sys 

#tienda = 'Tienda83'
tienda = sys.argv[1]
file = sys.argv[2]
old_data = pd.read_csv('data/model_data_'+ tienda + 'last.csv')
new_data = pd.read_csv('data/' + file)

new_data.FechaEntrega = pd.to_datetime(new_data.FechaEntrega)
new_data.FechaHoraLlegada = pd.to_datetime(new_data.FechaHoraLlegada)
new_data['Anio'] = new_data.FechaEntrega.dt.year
new_data['Mes'] = new_data.FechaEntrega.dt.month
new_data['Dia'] = new_data.FechaEntrega.dt.day
new_data['Hora'] = new_data.FechaHoraLlegada.dt.hour
new_data['FechaHora'] = new_data.FechaHoraLlegada.dt.strftime(r'%Y-%m-%d:%H')

#new_data = new_data[['FechaHora', 'Id_Solicitud_Entrega', 'TotalPaquetesEntregados', 'Anio', 'Mes', 'Dia', 'DiaSemana', 'Hora']]
new_data = new_data[['FechaHora', 'Id_Solicitud_Entrega', 'TotalPaquetesEntregados', 'Anio', 'Mes', 'Dia', 'Hora']]
new_data = new_data.groupby(['Anio','Mes', 'Dia','Hora']).agg(
    No_Clientes = ('Id_Solicitud_Entrega', 'nunique'), 
    Total_paquetes = ('TotalPaquetesEntregados', 'sum'),
    FechaHora = ('FechaHora', 'min'),    
    ).reset_index().sort_values(['FechaHora']).reset_index()
new_data = new_data[['FechaHora', 'Anio', 'Mes', 'Dia', 'Hora', 'Total_paquetes', 'No_Clientes']].sort_values('FechaHora')

old_data = old_data.append(new_data, ignore_index=True)

old_data.to_csv('data/model_data_' + tienda + 'last.csv', index=False)