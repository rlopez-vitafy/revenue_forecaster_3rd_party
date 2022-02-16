import streamlit as st
import numpy as np
import pandas as pd
import datetime

from io import BytesIO


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title='Revenue Forecasting ', layout='wide')

 
    
st.write("""
# Revenue Forecasting 3rd Party Decision Tree Algo

Über diese App wird für potenzielle Neulistungen der erwartete Umsatz für 3rd Party Produkte Produkte berechnet. :sunglasses:

Die Berechnung erfolgt dabei über die Attribute: """)


df_erklarung = pd.DataFrame(
    [
        ['brand', 'Markenname', 'Optimum Nutrition'],
        ['tag_primary', 'Primary Tag', 'Sport'],
        ['tag_secondary', 'Secondary Tag', 'Protein'],
        ['tag_tertiary', 'Tertiary Tag',  'Whey'],
        ['geschmack_filter', 'Hauptgeschmack', 'Schokolade'],
        ['product_dosage_form', 'Darreichungsform', 'Pulver'],
        ['VK', 'Der voraussichtliche Verkaufspreis', '47.99'],
        ['Sales Rank', 'Amazon Best Seller Rank', '12.555'],
        ['Potenzialfaktor', 'Eigene Performance-Einschätzung', 'Range 1-5, wobei 1=Topseller und 5=Penner'],
        ['qty sold Perzentil', 'Die monatlichen qty sold auf Basis des Potenzialfaktors', 'Perzentilgrenze im vergebenen Tertiary Tag']
    ],
    columns=['Attribut', 'Erklärung', 'Beispiel oder Info'])


st.write(df_erklarung)
#st.subheader("Beispiel")


df_OPS = pd.read_excel('OPS_avg_sales_3rd.xlsx', sheet_name = 'OPS')

#st.write(df_OPS[df_OPS.brand.eq('Optimum Nutrition')][:5])

    

X_OPS = df_OPS[['brand', 'tag_primary', 'tag_secondary', 'tag_tertiary', 'private_label_flag', 'geschmack_filter', 'product_dosage_form', 'vk_brutto', 'potenzialfaktor', 'qty sold perzentil', 'Sales Rank']]
y_OPS = df_OPS['umsatz_12to0w_avg2']


X_OPS_nb = df_OPS[['tag_primary', 'tag_secondary', 'tag_tertiary', 'private_label_flag', 'geschmack_filter', 'product_dosage_form', 'vk_brutto', 'potenzialfaktor', 'qty sold perzentil', 'Sales Rank']]
y_OPS_nb = df_OPS['umsatz_12to0w_avg2']



brand_list = df_OPS['brand'].unique()
prim_tag_list = df_OPS['tag_primary'].unique()
sec_tag_list = df_OPS['tag_secondary'].unique()
tert_tag_list = df_OPS['tag_tertiary'].unique()
pl_flag = df_OPS['private_label_flag'].unique()
flavour_list = df_OPS['geschmack_filter'].unique()
dosage_list = df_OPS['product_dosage_form'].unique()


st.subheader('Vorhersage')

status = st.radio('Auswahl Einzel-/ Mehrfachvorhersage: ', ('Einzelprognose', 'Mehrfachprognose'))

if status == 'Einzelprognose':
    
    status_brand = st.radio('Auswahl Brand: ', ('vorhandene Brand', 'neue Brand'))
    
    if status_brand == 'vorhandene Brand':
    
        parameter_brand = st.selectbox('Brand', brand_list)
        parameter_tag_prim = st.selectbox('tag primary', prim_tag_list)
        parameter_tag_sec = st.selectbox('tag secondary', sec_tag_list)
        parameter_tag_tert = st.selectbox('tag tertiary', tert_tag_list)
        parameter_pl = 0
        parameter_flavour = st.selectbox('Hauptgeschmack', flavour_list)
        parameter_dosage = st.selectbox('Darreichungsform', dosage_list)
        parameter_VK = st.number_input('VK', step=0.01)
        parameter_SalesRank = st.number_input('Sales Rank', step=1)
        parameter_Potenzialfaktor = st.number_input('Potenzialfaktor', min_value=1, max_value=5, step=1)
        parameter_QtySoldPerzentil = st.number_input('Qty Sold Perzentil', step=0.1)
    
    else:
        parameter_brand = st.text_input('Brand')
        parameter_tag_prim = st.selectbox('tag primary', prim_tag_list)
        parameter_tag_sec = st.selectbox('tag secondary', sec_tag_list)
        parameter_tag_tert = st.selectbox('tag tertiary', tert_tag_list)
        parameter_pl = 0
        parameter_flavour = st.selectbox('Hauptgeschmack', flavour_list)
        parameter_dosage = st.selectbox('Darreichungsform', dosage_list)
        parameter_VK = st.number_input('VK', step=0.01)
        parameter_SalesRank = st.number_input('Sales Rank', step=1)
        parameter_Potenzialfaktor = st.number_input('Potenzialfaktor', min_value=1, max_value=5, step=1)
        parameter_QtySoldPerzentil = st.number_input('Qty Sold Perzentil', step=0.1)
    
    
    
else:
    st.write(
             """
             Beachte:  
             Zum Upload nur die **erstellte Vorlage** benutzen.
             
             """
            )
    uploaded_file = st.file_uploader('', type=["xlsx"])
    if uploaded_file is not None:
       
        df_uploaded = pd.read_excel(uploaded_file, sheet_name = 'products')
        st.write("Hier die Vorschau deiner hochgeladenen Datei:")
        st.write(df_uploaded.head())
        
    status_brand_mehrfach = st.radio('Auswahl Brand: ', ('vorhandene Brand', 'neue Brand'))
        
    
    
    
start_prediction = st.button('Start the prediction')
if start_prediction:
    
    if status == 'Einzelprognose':
        
        if status_brand == 'vorhandene Brand':
            
        
            X_pred_einzel = [[parameter_brand, parameter_tag_prim, parameter_tag_sec, parameter_tag_tert, parameter_pl, parameter_flavour, parameter_dosage, parameter_VK, parameter_Potenzialfaktor, parameter_QtySoldPerzentil, parameter_SalesRank]]
            columns=['brand', 'tag_primary', 'tag_secondary', 'tag_tertiary', 'private_label_flag', 'geschmack_filter', 'product_dosage_form', 'vk_brutto', 'potenzialfaktor', 'qty sold perzentil', 'Sales Rank']

            df_pred_einzel = pd.DataFrame(X_pred_einzel, columns=columns)
            st.write(df_pred_einzel)

            cf = ColumnTransformer([
            ("brand", OneHotEncoder(), ["brand"]),
            ("tag_primary", OneHotEncoder(), ["tag_primary"]),
            ("tag_secondary", OneHotEncoder(), ["tag_secondary"]),
            ("tag_tertiary", OneHotEncoder(), ["tag_tertiary"]),
            ("geschmack_filter", OneHotEncoder(), ["geschmack_filter"]),
            ("product_dosage_form", OneHotEncoder(), ["product_dosage_form"])
            ], remainder = "passthrough"
            )
            cf.fit(X_OPS)
            X_transformed = cf.transform(X_OPS)


            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_OPS, train_size = 0.8)

            model_dtr = DecisionTreeRegressor(max_depth = 35,
                                      min_samples_split = 3,
                                      min_samples_leaf = 1,
                                      max_features= 'auto'
                                     )



            model_dtr.fit(X_train, y_train)

            st.subheader('Model Performance')
            st.write('Score train data: ')
            st.info(round(model_dtr.score(X_train, y_train),3))
            st.write('Score test data: ')
            st.info(round(model_dtr.score(X_test, y_test),3))

            y_pred = model_dtr.predict(X_transformed)
            df_m = df_OPS
            df_m[['Prediction']] = y_pred

            df_m.drop(df_m[df_m['Prediction'] == 0].index, inplace=True)


            def mape(df):
                df.drop(df[df['Prediction'] == 0].index, inplace=True)
                mape_value = np.mean(np.abs((df['umsatz_12to0w_avg2'] - df['Prediction']) / df['Prediction']))*100
                if math.isinf(mape_value):
                    return 0
                else:
                    return mape_value

            mape_series = df_m.groupby(['tag_tertiary']).apply(mape)
            df_mape = mape_series.to_frame(name='MAPE_value').reset_index()

            Mape_value = df_mape[df_mape['tag_tertiary'] == parameter_tag_tert][['MAPE_value']]


            prediction = model_dtr.predict(cf.transform(df_pred_einzel))

            df_output = df_pred_einzel[['brand', 'tag_tertiary']]
            df_output['prediction'] = prediction

            df_output_final = df_output.merge(df_mape, on = 'tag_tertiary', how = 'left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))


            st.write(df_output_final)
            
        else:
            
            X_pred_einzel = [[parameter_tag_prim, parameter_tag_sec, parameter_tag_tert, parameter_pl, parameter_flavour, parameter_dosage, parameter_VK, parameter_Potenzialfaktor, parameter_QtySoldPerzentil, parameter_SalesRank]]
            columns=['tag_primary', 'tag_secondary', 'tag_tertiary', 'private_label_flag', 'geschmack_filter', 'product_dosage_form', 'vk_brutto', 'potenzialfaktor', 'qty sold perzentil', 'Sales Rank']

            df_pred_einzel = pd.DataFrame(X_pred_einzel, columns=columns)
            st.write(df_pred_einzel)

            cf = ColumnTransformer([
            
            ("tag_primary", OneHotEncoder(), ["tag_primary"]),
            ("tag_secondary", OneHotEncoder(), ["tag_secondary"]),
            ("tag_tertiary", OneHotEncoder(), ["tag_tertiary"]),
            ("geschmack_filter", OneHotEncoder(), ["geschmack_filter"]),
            ("product_dosage_form", OneHotEncoder(), ["product_dosage_form"])
            ], remainder = "passthrough"
            )
            cf.fit(X_OPS_nb)
            X_transformed = cf.transform(X_OPS_nb)


            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_OPS_nb, train_size = 0.8)

            model_dtr = DecisionTreeRegressor(max_depth = 35,
                                      min_samples_split = 3,
                                      min_samples_leaf = 1,
                                      max_features= 'auto'
                                     )



            model_dtr.fit(X_train, y_train)

            st.subheader('Model Performance')
            st.write('Score train data: ')
            st.info(round(model_dtr.score(X_train, y_train),3))
            st.write('Score test data: ')
            st.info(round(model_dtr.score(X_test, y_test),3))

            y_pred = model_dtr.predict(X_transformed)
            df_m = df_OPS
            df_m[['Prediction']] = y_pred

            df_m.drop(df_m[df_m['Prediction'] == 0].index, inplace=True)


            def mape(df):
                df.drop(df[df['Prediction'] == 0].index, inplace=True)
                mape_value = np.mean(np.abs((df['umsatz_12to0w_avg2'] - df['Prediction']) / df['Prediction']))*100
                if math.isinf(mape_value):
                    return 0
                else:
                    return mape_value

            mape_series = df_m.groupby(['tag_tertiary']).apply(mape)
            df_mape = mape_series.to_frame(name='MAPE_value').reset_index()

            Mape_value = df_mape[df_mape['tag_tertiary'] == parameter_tag_tert][['MAPE_value']]


            prediction = model_dtr.predict(cf.transform(df_pred_einzel))
            
            #df_pred_einzel = pd.DataFrame(X_pred_einzel, columns=columns)

            values_output = [[parameter_brand, parameter_tag_tert, prediction]]
            columns_output = ['brand', 'tag_tertiary', 'prediction']
            
            df_output = pd.DataFrame(values_output, columns=columns_output)

            df_output_final = df_output.merge(df_mape, on = 'tag_tertiary', how = 'left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))


            st.write(df_output_final)
        
        
    if status == 'Mehrfachprognose':
        

        if status_brand_mehrfach == 'vorhandene Brand':
        
            X_pred_mehrfach = df_uploaded[['brand', 'tag_primary', 'tag_secondary', 'tag_tertiary', 'private_label_flag', 'geschmack_filter', 'product_dosage_form', 'vk_brutto', 'potenzialfaktor', 'qty sold perzentil', 'Sales Rank']]
            df_output = df_uploaded[['product_id', 'sku', 'brand', 'product_name' , 'tag_tertiary']]       

            cf = ColumnTransformer([
            ("brand", OneHotEncoder(), ["brand"]),
            ("tag_primary", OneHotEncoder(), ["tag_primary"]),
            ("tag_secondary", OneHotEncoder(), ["tag_secondary"]),
            ("tag_tertiary", OneHotEncoder(), ["tag_tertiary"]),
            ("geschmack_filter", OneHotEncoder(), ["geschmack_filter"]),
            ("product_dosage_form", OneHotEncoder(), ["product_dosage_form"])
            ], remainder = "passthrough"
            )
            cf.fit(X_OPS)
            X_transformed = cf.transform(X_OPS)


            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_OPS, train_size = 0.8)

            model_dtr = DecisionTreeRegressor(max_depth = 35,
                                      min_samples_split = 3,
                                      min_samples_leaf = 1,
                                      max_features= 'auto'
                                     )



            model_dtr.fit(X_train, y_train)

            st.subheader('Model Performance')
            st.write('Score train data: ')
            st.info(round(model_dtr.score(X_train, y_train),3))
            st.write('Score test data: ')
            st.info(round(model_dtr.score(X_test, y_test),3))

            y_pred = model_dtr.predict(X_transformed)

            df_m = df_OPS
            df_m[['Prediction']] = y_pred

            df_m.drop(df_m[df_m['Prediction'] == 0].index, inplace=True)



            def mape(df):
                df.drop(df[df['Prediction'] == 0].index, inplace=True)
                mape_value = np.mean(np.abs((df['umsatz_12to0w_avg2'] - df['Prediction']) / df['Prediction']))*100
                if math.isinf(mape_value):
                    return 0
                else:
                    return mape_value

            mape_series = df_m.groupby(['tag_tertiary']).apply(mape)


            df_mape = mape_series.to_frame(name='MAPE_value').reset_index()        


            prediction = model_dtr.predict(cf.transform(X_pred_mehrfach))
            df_output['prediction'] = prediction


            df_output_final = df_output.merge(df_mape, on = 'tag_tertiary', how = 'left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))

            st.write(df_output_final)

            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine="xlsxwriter")
                df.to_excel(writer, sheet_name='Sheet1')
                writer.save()
                processed_data = output.getvalue()
                return processed_data


            df_xlsx = to_excel(df_output_final)

            date_today = str(datetime.date.today())
            st.download_button('Ergebnisse als Excel herunterladen', 
                               df_xlsx,
                               file_name = date_today+'_forecast-prediction.xlsx'
                               )
        
        
        else:
            
            
            X_pred_mehrfach = df_uploaded[['tag_primary', 'tag_secondary', 'tag_tertiary', 'private_label_flag', 'geschmack_filter', 'product_dosage_form', 'vk_brutto', 'potenzialfaktor', 'qty sold perzentil', 'Sales Rank']]
            df_output = df_uploaded[['product_id', 'sku', 'brand', 'product_name' , 'tag_tertiary']]       

            cf = ColumnTransformer([

            ("tag_primary", OneHotEncoder(), ["tag_primary"]),
            ("tag_secondary", OneHotEncoder(), ["tag_secondary"]),
            ("tag_tertiary", OneHotEncoder(), ["tag_tertiary"]),
            ("geschmack_filter", OneHotEncoder(), ["geschmack_filter"]),
            ("product_dosage_form", OneHotEncoder(), ["product_dosage_form"])
            ], remainder = "passthrough"
            )
            cf.fit(X_OPS_nb)
            X_transformed = cf.transform(X_OPS_nb)


            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_OPS_nb, train_size = 0.8)

            model_dtr = DecisionTreeRegressor(max_depth = 35,
                                      min_samples_split = 3,
                                      min_samples_leaf = 1,
                                      max_features= 'auto'
                                     )



            model_dtr.fit(X_train, y_train)

            st.subheader('Model Performance')
            st.write('Score train data: ')
            st.info(round(model_dtr.score(X_train, y_train),3))
            st.write('Score test data: ')
            st.info(round(model_dtr.score(X_test, y_test),3))

            y_pred = model_dtr.predict(X_transformed)

            df_m = df_OPS
            df_m[['Prediction']] = y_pred

            df_m.drop(df_m[df_m['Prediction'] == 0].index, inplace=True)



            def mape(df):
                df.drop(df[df['Prediction'] == 0].index, inplace=True)
                mape_value = np.mean(np.abs((df['umsatz_12to0w_avg2'] - df['Prediction']) / df['Prediction']))*100
                if math.isinf(mape_value):
                    return 0
                else:
                    return mape_value

            mape_series = df_m.groupby(['tag_tertiary']).apply(mape)


            df_mape = mape_series.to_frame(name='MAPE_value').reset_index()        


            prediction = model_dtr.predict(cf.transform(X_pred_mehrfach))
            df_output['prediction'] = prediction


            df_output_final = df_output.merge(df_mape, on = 'tag_tertiary', how = 'left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))

            st.write(df_output_final)

            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine="xlsxwriter")
                df.to_excel(writer, sheet_name='Sheet1')
                writer.save()
                processed_data = output.getvalue()
                return processed_data


            df_xlsx = to_excel(df_output_final)

            date_today = str(datetime.date.today())
            st.download_button('Ergebnisse als Excel herunterladen', 
                               df_xlsx,
                               file_name = date_today+'_forecast-prediction.xlsx'
                               )
    