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

# ... (your code continues)

start_prediction = st.button('Start the prediction')
if start_prediction:
    if status == 'Einzelprognose':
        if status_brand == 'vorhandene Brand':
            # ... (your existing code)

            print("Original data:")
            print(df_pred_einzel)

            prediction = model_dtr.predict(cf.transform(df_pred_einzel))

            print("Predicted values:")
            print(prediction)

            df_output = df_pred_einzel[['brand', 'tag_tertiary']]
            df_output['prediction'] = prediction

            df_output_final = df_output.merge(df_mape, on='tag_tertiary', how='left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))

            st.write(df_output_final)

        else:
            # ... (your existing code)

            print("Original data:")
            print(df_pred_einzel)

            prediction = model_dtr.predict(cf.transform(df_pred_einzel))

            print("Predicted values:")
            print(prediction)

            df_output = df_pred_einzel[['brand', 'tag_tertiary']]
            df_output['prediction'] = prediction

            df_output_final = df_output.merge(df_mape, on='tag_tertiary', how='left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))

            st.write(df_output_final)

    elif status == 'Mehrfachprognose':
        if status_brand_mehrfach == 'vorhandene Brand':
            # ... (your existing code)

            print("Original data:")
            print(X_pred_mehrfach)

            prediction = model_dtr.predict(cf.transform(X_pred_mehrfach))
            df_output['prediction'] = prediction

            df_output_final = df_output.merge(df_mape, on='tag_tertiary', how='left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))

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
                               file_name=date_today + '_forecast-prediction.xlsx'
                               )

        else:
            # ... (your existing code)

            print("Original data:")
            print(X_pred_mehrfach)

            prediction = model_dtr.predict(cf.transform(X_pred_mehrfach))
            df_output['prediction'] = prediction

            df_output_final = df_output.merge(df_mape, on='tag_tertiary', how='left')

            df_output_final['lower limit'] = df_output_final['prediction'] - (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))
            df_output_final['upper limit'] = df_output_final['prediction'] + (
                    df_output_final['prediction'] * (df_output_final['MAPE_value'] / 100))

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
                               file_name=date_today + '_forecast-prediction.xlsx'
                               )
