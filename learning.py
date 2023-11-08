import os
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor


# 이 파일은 텐서플로우를 이용하여 시계열 학습을 진행합니다.
# 하단에 종종 보이는 matplotlib의 pyplot은 전처리 과정과 loss 값 확인을 위해 있었습니다.

# 디렉토리 내의 모든 CSV 파일 목록을 가져옵니다.
directory_path = "C:/Users/mycom/Desktop/db_csv"  # 디렉토리 경로를 수정하세요.
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# 예측할 날짜를 설정합니다 (예: 12월 31일).
prediction_date = "2023-12-31" 

#텐서플로우 GPU로 학습하기 위한 코드입니다.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가를 허용 (메모리 사용량에 따라 조절)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

for file in csv_files:
    # CSV 파일을 읽어옵니다.
    print(file)
    stock_data = pd.read_csv(os.path.join(directory_path, file), encoding='cp949')

    # 데이터 전처리
    # save original 'Open' prices for later
    original_open = stock_data['close'].values

    # separate dates for future plotting
    dates = pd.to_datetime(stock_data['date'])

    # variables for training
    # cols = list(stock_data)[['open','close','high','low','volume']]  # 수정: column 이름 수정
    cols = ['open', 'close', 'high', 'low', 'volume']
    # new dataframe with only training data - 5 columns
    stock_data = stock_data[cols].astype(float)

    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(stock_data)
    stock_data_scaled = scaler.transform(stock_data)

    # split to train data and test data
    n_train = int(0.9 * stock_data_scaled.shape[0])
    train_data_scaled = stock_data_scaled[0: n_train]
    train_dates = dates[0: n_train]

    test_data_scaled = stock_data_scaled[n_train:]
    test_dates = dates[n_train:]

    # data reformatting for LSTM
    pred_days = 1  # prediction period
    seq_len = 14  # sequence length = past days for future prediction.
    input_dim = 5  # input_dimension = ['Open', 'High', 'Low', 'Close', 'Volume']

    trainX = []
    trainY = []
    testX = []
    testY = []

    for i in range(seq_len, n_train - pred_days + 1):
        trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    # LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(trainY.shape[1]))

    model.summary()

    # specify your learning rate
    learning_rate = 0.01
    # create an Adam optimizer with the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    # compile your model using the custom optimizer
    model.compile(optimizer=optimizer, loss='mse', run_eagerly=True)

    # Try to load weights
    try:
        model.load_weights('./save_weights/lstm_weights.h5')
        print("Loaded model weights from disk")
    except:
        print("No weights found, training model from scratch")
        # Fit the model
        history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
        # Save model weights after training
        #model.save_weights('./save_weights/lstm_weights.h5')
        model.save_weights('./lstm_weights.h5')

        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        #plt.legend()
        #plt.show()

    # prediction
    prediction = model.predict(testX)
    print(prediction.shape, testY.shape)

    # generate array filled with means for prediction
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

    # substitute predictions into the first column
    mean_values_pred[:, 0] = np.squeeze(prediction)

    # inverse transform
    y_pred = scaler.inverse_transform(mean_values_pred)[:, 0]
    print(y_pred.shape)

    # generate array filled with means for testY
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

    # substitute testY into the first column
    mean_values_testY[:, 0] = np.squeeze(testY)

    # inverse transform
    testY_original = scaler.inverse_transform(mean_values_testY)[:, 0]
    print(testY_original.shape)

    # plotting
    plt.figure(figsize=(14, 5))

    # plot original 'Open' prices
    plt.plot(dates, original_open, color='green', label='Original Open Price')

    # plot actual vs predicted
    plt.plot(test_dates[seq_len:], testY_original, color='blue', label='Actual Open Price')
    plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label='Predicted Open Price')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Original, Actual and Predicted Open Price')
    #plt.legend()
    #plt.show()

    # Calculate the start and end indices for the zoomed plot
    zoom_start = len(test_dates) - 50
    zoom_end = len(test_dates)

    # Create the zoomed plot
    plt.figure(figsize=(14, 5))

    # Adjust the start index for the testY_original and y_pred arrays
    adjusted_start = zoom_start - seq_len

    plt.plot(test_dates[zoom_start:zoom_end],
             testY_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
             color='blue',
             label='Actual Open Price')

    plt.plot(test_dates[zoom_start:zoom_end],
             y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start],
             color='red',
             linestyle='--',
             label='Predicted Open Price')

    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Zoomed In Actual vs Predicted Open Price')
    #plt.legend()
    #plt.show()

    # 예측값을 원본 데이터 아래에 추가합니다.
    #stock_data['Predicted_Open'] = y_pred  # 예측값에 해당하는 변수명 사용

    # 파일명을 수정하여 새로운 CSV 파일로 저장합니다.
    predicted_data = pd.DataFrame({'predicted_close': y_pred})
    # 새로운 폴더 경로를 지정
    output_folder = 'C:/Users/mycom/Desktop/predict'  # 원하는 출력 폴더 경로로 수정
    # 원래 파일명에서 확장자 제거
    file_name_without_extension = os.path.splitext(os.path.basename(file))[0]
    # 새로운 CSV 파일 경로 생성
    new_file_path = os.path.join(output_folder, f"{file_name_without_extension}_predict.csv")
    # 예측 데이터를 새로운 CSV 파일로 저장
    predicted_data.to_csv(new_file_path, index=False)
    print(f"Predictions for {file} saved to {new_file_path}")

# 모든 CSV 파일에 대한 예측이 완료되면 종료합니다.
# GPU를 이용해서 학습을 진행하기 위해 멀티 스레드를 사용하지 않아 약 48시간 이상이 경과하였습니다..