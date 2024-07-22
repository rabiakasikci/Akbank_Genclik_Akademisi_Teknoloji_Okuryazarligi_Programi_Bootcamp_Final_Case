# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:29:15 2024

@author: Rabia KAŞIKCI
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:45:11 2024

@author: Rabia KAŞIKCI
"""

# Import Libraries
import random
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import yfinance as yf
import time
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from PIL import Image, ImageTk, ImageSequence
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.callbacks import EarlyStopping

# BIST 30 Stock List
bist_30_list = [
    "AKBNK", "ALARK", "ASELS", "BIMAS", "BRSAN", "EKGYO",
    "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS",
    "ISCTR", "KCHOL", "KOZAL", "KRDMD", "OYAKC", "PETKM",
    "PGSUS", "SAHOL", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"
]

def bist30_info():
    try:
        # Create new window for stock information
        info_window = tk.Toplevel(root)
        info_window.title("BİST30 Information")
        info_window.geometry("700x500")
        info_window.configure(bg='lightyellow')
        
        label = tk.Label(
            info_window,
            text=(
                "Below you can see the list of stocks I can predict.\n"
                "Sembol  Company\n"
                "AKBNK - Akbank T.A.Ş.\n"
                "ALARK - Alarko Holding A.Ş.\n"
                "ASELS - Aselsan Elektronik Sanayi ve Ticaret A.Ş.\n"
                "BIMAS - BİM Birleşik Mağazalar A.Ş.\n"
                "BRSAN - Borusan Mannesmann Boru Sanayi ve Ticaret A.Ş.\n"
                "EKGYO - Emlak Konut Gayrimenkul Yatırım Ortaklığı A.Ş.\n"
                "ENKAI - Enka İnşaat ve Sanayi A.Ş.\n"
                "EREGL - Ereğli Demir ve Çelik Fabrikaları T.A.Ş.\n"
                "FROTO - Ford Otomotiv Sanayi A.Ş.\n"
                "GARAN - Türkiye Garanti Bankası A.Ş.\n"
                "GUBRF - Gübre Fabrikaları T.A.Ş.\n"
                "HEKTS - Hektaş Ticaret T.A.Ş.\n"
                "ISCTR - Türkiye İş Bankası A.Ş. (C)\n"
                "KCHOL - Koç Holding A.Ş.\n"
                "KOZAL - Koza Altın İşletmeleri A.Ş.\n"
                "KRDMD - Kardemir Karabük Demir Çelik Sanayi ve Ticaret A.Ş. (D)\n"
                "OYAKC - OYAK Çimento Fabrikaları A.Ş.\n"
                "PETKM - Petkim Petrokimya Holding A.Ş.\n"
                "PGSUS - Pegasus Hava Taşımacılığı A.Ş.\n"
                "SAHOL - Hacı Ömer Sabancı Holding A.Ş.\n"
                "TCELL - Turkcell İletişim Hizmetleri A.Ş.\n"
                "THYAO - Türk Hava Yolları A.O.\n"
                "TOASO - Tofaş Türk Otomobil Fabrikası A.Ş.\n"
                "TUPRS - Türkiye Petrol Rafinerileri A.Ş. (Tüpraş)\n"
                "YKBNK - Yapı ve Kredi Bankası A.Ş.\n"
            ),
            
        )
        label.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", str(e))
    
    

def fetch_stock_data(symbol):
    prices_train = yf.download(symbol, start="2014-01-31", end="2024-12-31")
    if prices_train.empty:
        raise ValueError("No data found for this stock.")
    prices_train.reset_index(inplace=True)
    prices_train['Date'] = pd.to_datetime(prices_train['Date'])
    prices_train.fillna(prices_train.mean(), inplace=True)
    return prices_train

def stock_info():
    try:
        user_input = entry.get().upper() + ".IS"
        prices_train = fetch_stock_data(user_input)

     
        
        
        plt.style.use('dark_background')
        # Creating the figure and axis objects
        fig, ax = plt.subplots(figsize=(14, 7))

        # Setting the face color to black
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')

        # Plotting the data
        ax.plot(prices_train['Date'], prices_train['Open'], label='Open Value', color='cyan')
        ax.plot(prices_train['Date'], prices_train['High'], label='High Value', color='red')
        ax.plot(prices_train['Date'], prices_train['Low'], label='Low Value', color='blue')
        ax.plot(prices_train['Date'], prices_train['Adj Close'], label='Adj Close', color='yellow')
        ax.plot(prices_train['Date'], prices_train['Close'], label='Close Value', color='green')

        # Setting titles and labels with white color
        ax.set_title(f'{user_input} - All Value', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Value', color='white')

        # Setting legend with appropriate face and edge colors
        legend = ax.legend(facecolor='black', edgecolor='white', loc='best')
        for text in legend.get_texts():
            text.set_color('white')

        # Setting ticks and grid with white color
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, linestyle='--', color='gray', alpha=0.6)

        # Ensuring layout is tight
        plt.tight_layout()

        # Saving the figure
        plt.savefig("open_info_plot.png")

        # Closing the figure
        plt.close()
        
        
        
        plt.style.use('dark_background')
        # Creating the figure and axis objects
        fig, ax = plt.subplots(figsize=(14, 7))
        # Setting the face color to black
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')
        # Plotting the bar chart for volume values
        ax.bar(prices_train['Date'], prices_train['Volume'], label='Volume Value', color='cyan')
        # Setting titles and labels with white color
        ax.set_title(f'{user_input} - Volume Value', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Volume Value', color='white')
        # Setting legend with appropriate face and edge colors
        legend = ax.legend(facecolor='black', edgecolor='white', loc='best')
        for text in legend.get_texts():
            text.set_color('white')

        # Setting ticks and grid with white color
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, linestyle='--', color='gray', alpha=0.6)
        # Ensuring layout is tight
        plt.tight_layout()
        # Saving the figure
        plt.savefig("volume_info_plot.png")
        # Closing the figure
        plt.close()







        
        
        # Create new window for stock information
        info_window = tk.Toplevel(root)
        info_window.title("Stock Information")
        info_window.geometry("700x1000")
        info_window.configure(bg='lightyellow')

        info_label = tk.Label(info_window, text=f"{user_input} stock information:\n", bg='lightyellow')
        info_label.pack(pady=10)

      
        # Display Opening Values Plot
        opening_plot_image = Image.open("open_info_plot.png")
        opening_plot_image = opening_plot_image.resize((700, 300), Image.ANTIALIAS)
        
        opening_plot_photo = ImageTk.PhotoImage(opening_plot_image)
        opening_plot_label = tk.Label(info_window, image=opening_plot_photo, bg='lightyellow')
        opening_plot_label.image = opening_plot_photo  # Keep reference
        opening_plot_label.pack(pady=10)
        
        
        # Display Opening Values Plot
        opening_plot_image = Image.open("volume_info_plot.png")
        opening_plot_image = opening_plot_image.resize((700, 300), Image.ANTIALIAS)
        opening_plot_photo = ImageTk.PhotoImage(opening_plot_image)
        opening_plot_label = tk.Label(info_window, image=opening_plot_photo, bg='red')
        opening_plot_label.image = opening_plot_photo  # Keep reference
        opening_plot_label.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def stock_prediction():
    entry_label.config(text="I'm doing some calculations. Can you hold on a second?")
    root.update() 
    try:
        
        user_input = entry.get().upper() + ".IS"
        prices_train = fetch_stock_data(user_input)
        
        # Show message before calculations
    
        #prices_train = yf.download("AKBNK.IS", start="2024-01-31", end="2024-12-31")
        if prices_train.empty:
            raise ValueError("No data found for this stock.")


        # Veriyi işleme
        prices_train.reset_index(inplace=True)
        prices_train['Date'] = pd.to_datetime(prices_train['Date'])
        prices_train.fillna(prices_train.mean(), inplace=True)
       

        last_year = prices_train['Date'].max().year
        last_year_data = prices_train[prices_train['Date'].dt.year == last_year]
        prices_train_without_last_year = prices_train[~prices_train['Date'].isin(last_year_data['Date'])]

        # Eğitim ve test ayrımı
        train, test = train_test_split(prices_train_without_last_year, test_size=0.3, shuffle=False) # Zaman serisi için karıştırma yok
        x_train = train[['Open', 'High', 'Low', 'Volume']].values
        x_test = test[['Open', 'High', 'Low', 'Volume']].values
        y_train = train['Close'].values
        y_test = test['Close'].values

        # Doğrusal Regresyon
        model_lnr = LinearRegression()
        model_lnr.fit(x_train, y_train)
        y_pred_lr = model_lnr.predict(x_test)

        MSE_lr = round(mean_squared_error(y_test, y_pred_lr), 3)
        RMSE_lr = round(np.sqrt(MSE_lr), 3)
        MAE_lr = round(mean_absolute_error(y_test, y_pred_lr), 3)
        MAPE_lr = round(mean_absolute_percentage_error(y_test, y_pred_lr), 3)
        R2_Score_lr = round(r2_score(y_test, y_pred_lr), 3)

        # ARIMA Modeli
        close_value_arima = prices_train_without_last_year['Close']
        p_value = adfuller(close_value_arima)[1]
        if p_value > 0.05:
            close_value_arima = close_value_arima.diff().dropna()

        model_arima = auto_arima(close_value_arima, seasonal=False, stepwise=True, trace=True)
        model_fit = model_arima.fit(close_value_arima)
        forecast = model_fit.predict(n_periods=10)

        y_test_arima = prices_train_without_last_year['Close'][-10:]
        MSE_arima = round(mean_squared_error(y_test_arima, forecast), 3)
        RMSE_arima = round(np.sqrt(MSE_arima), 3)
        MAE_arima = round(mean_absolute_error(y_test_arima, forecast), 3)
        MAPE_arima = round(mean_absolute_percentage_error(y_test_arima, forecast), 3)
        R2_Score_arima = round(r2_score(y_test_arima, forecast), 3)

        # LSTM Modeli
        data_lstm = prices_train['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_lstm)

        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 10
        X, y = create_dataset(scaled_data, time_step)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model_lstm = Sequential()
        model_lstm.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(LSTM(100, return_sequences=False))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(Dense(50))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        history = model_lstm.fit(X_train, y_train, batch_size=32, epochs=80, validation_split=0.1, verbose=1, 
                                callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

        predictions = model_lstm.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        MSE_lstm = mean_squared_error(y_test, predictions)
        MAE_lstm = mean_absolute_error(y_test, predictions)
        RMSE_lstm = np.sqrt(MSE_lstm)
        R2_Score_lstm = r2_score(y_test, predictions)

        # En iyi modeli seçme
        min_mse = min(MSE_lr, MSE_arima, MSE_lstm)
        if min_mse == MSE_lr:
            y_pred_last_year = model_lnr.predict(last_year_data[['Open', 'High', 'Low', 'Volume']].values)
            model_name = 'Doğrusal Regresyon'
        elif min_mse == MSE_arima:
            close_value_arima_last_year = last_year_data['Close']
            p_value = adfuller(close_value_arima_last_year)[1]
            if p_value > 0.05:
                close_value_arima_last_year = close_value_arima_last_year.diff().dropna()
            model_arima_last_year = auto_arima(close_value_arima_last_year, seasonal=False, stepwise=True, trace=True)
            model_fit_last_year = model_arima_last_year.fit(close_value_arima_last_year)
            y_pred_last_year = model_fit_last_year.predict(n_periods=len(last_year_data))
            model_name = 'ARIMA'
        else:
            scaled_last_year = scaler.transform(last_year_data['Close'].values.reshape(-1, 1))
            X_last_year, _ = create_dataset(scaled_last_year, time_step)
            X_last_year = X_last_year.reshape(X_last_year.shape[0], X_last_year.shape[1], 1)
            y_pred_last_year = model_lstm.predict(X_last_year)
            y_pred_last_year = scaler.inverse_transform(y_pred_last_year)
            model_name = 'LSTM'

        # Grafik çizimi
        plt.style.use('dark_background')  # Beyaz arka plan tercihiniz varsa bu satırı değiştirebilirsiniz
        plt.figure(figsize=(14, 7))
        plt.plot(last_year_data['Date'], last_year_data['Close'], label='True Value', color='cyan')
        plt.plot(last_year_data['Date'][:len(y_pred_last_year)], y_pred_last_year, label=f'Pred ({model_name})', color='magenta')
        plt.title(f'True and Pred Close Value ({model_name})', color='white')
        plt.xlabel('Date', color='white')
        plt.ylabel('Close Value', color='white')
        plt.legend(facecolor='black', edgecolor='white', loc='best')
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        plt.grid(True, linestyle='--', color='gray', alpha=0.6)
        plt.tight_layout()

        plt.savefig("prediction_plot.png")
        plt.show()

        print(f'En İyi Modelin MSE: {min_mse}')



        
        

        """--------------------Future Prediction ---------------------------"""
        #ETS 
        # Modeli oluştur
        prices_train.set_index('Date', inplace=True)

        # Modeli oluştur
        model_ets = ExponentialSmoothing(prices_train['Close'], trend=None, seasonal='mul', seasonal_periods=30)
        results_ets = model_ets.fit()
        forecast_steps = 10
        forecast_ets = results_ets.forecast(steps=forecast_steps)

        # Son 10 gün için gerçek ve tahmin edilen değerleri karşılaştır
        y_true = prices_train['Close'][-forecast_steps:]
        y_pred = forecast_ets.values

        # İndeksleri oluşturma
        forecast_index = pd.date_range(start=prices_train.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

        # Karşılaştırma için DataFrame oluşturma
        comparison_df = pd.DataFrame({'True': y_true.values, 'Predicted': y_pred}, index=prices_train.index[-forecast_steps:])
        print(comparison_df)

        # Performans metriklerini hesapla
        mse_ets = mean_squared_error(y_true, y_pred)
        mae_ets = mean_absolute_error(y_true, y_pred)
        rmse_ets = np.sqrt(mse_ets)
        r2_ets = r2_score(y_true, y_pred)

        print(f'Mean Squared Error (MSE): {mse_ets}')
        print(f'Mean Absolute Error (MAE): {mae_ets}')
        print(f'Root Mean Squared Error (RMSE): {rmse_ets}')
        print(f'R^2 Score: {r2_ets}')

        # Yalnızca kapanış fiyatlarını kullan
        data = prices_train['Close'].values.reshape(-1, 1)

        # Veriyi ölçeklendir
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # LSTM giriş verilerini oluştur
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 10
        X, y = create_dataset(scaled_data, time_step)

        # Veriyi eğitim ve test setlerine böl
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # LSTM modeline uygun hale getirmek için veriyi yeniden şekillendir
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM modelini oluştur
        model_lstm = Sequential()
        model_lstm.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(LSTM(100, return_sequences=False))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(Dense(50))
        model_lstm.add(Dense(1))

        # Modeli derle
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Modeli eğit
        history = model_lstm.fit(X_train, y_train, batch_size=32, epochs=80, validation_split=0.1, verbose=1, 
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

        # Tahmin yap
        predictions = model_lstm.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Gerçek değerleri yeniden ölçeklendir
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Performans metriklerini hesapla
        mse_lstm = mean_squared_error(y_test, predictions)
        mae_lstm = mean_absolute_error(y_test, predictions)
        rmse_lstm = np.sqrt(mse_lstm)
        r2_lstm = r2_score(y_test, predictions)

        print(f'Mean Squared Error (MSE): {mse_lstm}')
        print(f'Mean Absolute Error (MAE): {mae_lstm}')
        print(f'Root Mean Squared Error (RMSE): {rmse_lstm}')
        print(f'R^2 Score: {r2_lstm}')



        if mse_ets < mse_lstm :
            
            # Visualize the forecast
            plt.figure(figsize=(12, 6))
            plt.plot(prices_train.index, prices_train['Close'], label='Gerçek Veri')
            plt.plot(forecast_index, forecast_ets, color='red', linestyle='--', label='Tahmin')
            plt.title('Future Prediction -ETS ')
            plt.xlabel('Date')
            plt.ylabel('Close')
            
            plt.xlim([pd.Timestamp('2023-01-01'), forecast_index[-1] + pd.Timedelta(days=5)])
            plt.savefig("future_pred.png")
            plt.legend()
            plt.show()
        else :
            last_60_days = scaled_data[-10:]
            predictions_last10_lstm = []

            # 10 gün boyunca tahmin yapın
            for _ in range(10):
                X_predict = np.array([last_60_days]).reshape(1, time_step, 1)
                next_day_prediction = model_lstm.predict(X_predict)
                predictions_last10_lstm.append(next_day_prediction[0, 0])  # İlk elemanı alın
                # Tahmin edilen değeri ekleyin ve eski verileri kaydırın
                last_60_days = np.append(last_60_days[1:], next_day_prediction, axis=0)

            # Tahminleri ters ölçekleyin
            predictions_last10_lstm = scaler.inverse_transform(np.array(predictions_last10_lstm).reshape(-1, 1))
            
            
            
            print("Gelecek 10 günün tahminleri:", predictions_last10_lstm)
            # Visualize the forecast
            plt.figure(figsize=(12, 6))
            plt.plot(prices_train.index, prices_train['Close'], label='Gerçek Veri')
            plt.plot(forecast_index, predictions_last10_lstm, color='red', linestyle='-', label='Tahmin')
            plt.title('Future Prediction -LSTM ')
            plt.xlabel('Date')
            plt.ylabel('Close')
            plt.xlim([pd.Timestamp('2023-01-01'), forecast_index[-1] + pd.Timedelta(days=5)])


            
            
            plt.savefig("future_pred.png")
            plt.legend()
            plt.show()
        
        

        # Create new window for stock information
        info_window = tk.Toplevel(root)
        info_window.title("Stock Information")
        info_window.geometry("700x1000")
        info_window.configure(bg='lightyellow')

        # Display information label
        #text = f"{user_input.upper()} stock information:" +"\n" + 
        info_label = tk.Label(info_window, text=f"{user_input.upper()} stock information:\n The first graph gives the output of the best algorithm in the prediction results using Open,High,Low,Volume. \n The second graph gives the output of the graph that makes the best prediction for 10 days from today.", bg='lightyellow')
        info_label.pack(pady=10)

        # Display stock information
        stock_info = f"{user_input.upper()} stock prediction"
        info_text = tk.Label(info_window, text=stock_info, bg='lightyellow')
        info_text.pack(pady=10)

        # Display plot image
        plot_image = Image.open("prediction_plot.png")
        plot_image = plot_image.resize((700, 300), Image.ANTIALIAS)
        plot_photo = ImageTk.PhotoImage(plot_image)
        plot_label = tk.Label(info_window, image=plot_photo, bg='lightyellow')
        plot_label.image = plot_photo  # Keep reference
        plot_label.pack(pady=10)
        
        #Display sec plot
        plot_image2 = Image.open("future_pred.png")
        plot_image2 = plot_image2.resize((700, 300), Image.ANTIALIAS)
        plot_photo2 = ImageTk.PhotoImage(plot_image2)
        plot_label2 = tk.Label(info_window, image=plot_photo2, bg='lightyellow')
        plot_label2.image = plot_photo2  # Keep a reference to avoid garbage collection
        plot_label2.pack(pady=10)
    

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        entry_label.config(text="The predictions are finalised. You can view the results. \n If you want to enter another stock, just write its symbol")
        root.update()

# Create the main window
root = tk.Tk()
root.title("Stock Prediction")
root.geometry("800x600")
root.configure(bg='white')

"""
# Add background image
background_image = Image.open("chatbot.gif")
background_image = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)
"""
# Load and display GIF
gif_path = 'imm.gif'  # Replace with your GIF file path
gif_image = Image.open(gif_path)

frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(gif_image)]
label_gif = tk.Label(root)
label_gif.pack()

def update_gif(index):
    frame = frames[index]
    label_gif.config(image=frame)
    root.after(100, update_gif, (index + 1) % len(frames))

update_gif(0)



# Create and pack widgets
entry_label = tk.Label(root, text="Enter Stock Symbol (e.g., 'AAPL'):", bg='lightyellow')
entry_label.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=10)

info_button = tk.Button(root, text="Show Stock Information", command=stock_info)
info_button.pack(pady=10)

pred_button = tk.Button(root, text="Show Stock Predictions", command=stock_prediction)
pred_button.pack(pady=10)

liste_button = tk.Button(root, text="Show Bist30 List", command=bist30_info)
liste_button.pack(pady=10)





root.mainloop()




























# Add text entry box
entry = tk.Entry(root)
entry.pack(pady=5)

# Add a button
button = tk.Button(root, text="Show stock prediction!", command=check_text)
button.pack(pady=5)

# Start the main loop
root.mainloop()
