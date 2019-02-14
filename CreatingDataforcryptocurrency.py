import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random

SEQLEN=60
RATIO_TO_PREDICT="LTC-USD"
FUTURE_PERIOD_PREDICT=3

def classify(current,future):
    if(float(future)>float(current)):
        return 1
    else:
        return 0

def preprocess_df(df):
    df.drop(columns=['future'],inplace=True)
    scaler=preprocessing.MinMaxScaler()
    for col in df.columns:
        if(col!='target'):
            df[col]=df[col].pct_change()
            df.dropna(inplace=True)
            df.fillna(method='ffill',inplace=True)
            df[col] = preprocessing.scale(pd.DataFrame(df[col].values))
            print(df[col].head())
    df.dropna(inplace=True)
    df.fillna(method='ffill',inplace=True)
    sequential_data=[]
    prev_days=deque(maxlen=SEQLEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if(len(prev_days)==SEQLEN):
            sequential_data.append([np.array(prev_days),i[-1]])

    random.shuffle(sequential_data)

    buys=[]
    sells=[]
    for seq,target in sequential_data:
        if(target==0):
            sells.append([seq,target])
        elif(target==1):
            buys.append([seq,target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower=min(len(buys),len(sells))
    buys=buys[:lower]
    sells=sells[:lower]
    sequential_data=buys+sells
    random.shuffle(sequential_data)
    X=[]
    Y=[]
    for seq,target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X),Y

ratios=["BTC-USD","LTC-USD","BCH-USD","ETH-USD"]
main_df=pd.DataFrame()
for ratio in ratios:
    dataset=f'D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/crypto_data/{ratio}.csv'
    df=pd.read_csv(dataset,names=["time","low","high","open","close","volume"])
    df.rename(columns={"close":f"{ratio}_close","volume":f"{ratio}_volume"},inplace=True)
    df.set_index("time",inplace=True)
    df=df[[f"{ratio}_close",f"{ratio}_volume"]]
    if(len(main_df)==0):
        main_df=df
    else:
        main_df=main_df.join(df)

main_df.fillna(method='ffill',inplace=True) #to fill empty values with previous values
main_df.dropna(inplace=True)
main_df['future']=main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target']=list(map(classify,main_df[f"{RATIO_TO_PREDICT}_close"],main_df['future']))
#print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]].head())
main_df.dropna(inplace=True)
main_df.fillna(method='ffill',inplace=True)
times=sorted(main_df.index.values)
last_5pct=times[-int(0.05*len(times))]
validation_main_df=main_df[(main_df.index>=last_5pct)]
main_df=main_df[(main_df.index<last_5pct)]

train_x,train_y=preprocess_df(main_df)
validation_x,validation_y=preprocess_df(validation_main_df)

print("Training Feature len = ",len(train_x)," Training Label Len = ",len(train_y))
print("Testing Feature len = ",len(validation_x)," Testing Label Len = ",len(validation_y))

np.save('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/train_x.npy',train_x)
np.save('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/train_y.npy',train_y)
np.save('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/validation_x.npy',validation_x)
np.save('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/validation_y.npy',validation_y)

    
