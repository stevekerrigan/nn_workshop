import fredapi
import pandas as pd
import numpy as np
import pickle

fred=fredapi.Fred(api_key_file=".fred_api_key") # You will need to register for this through st. louis fred.
train_data=fred.get_series("UNRATE",observation_start="1960-01-01",observation_end="1997-01-01")
test_data=fred.get_series("UNRATE",observation_start="1997-04-01",observation_end="2015-12-01")
horiz=0
def make_data(x):
    x=x/100
    x=pd.DataFrame(x)
    def differ(x,y):
        if y>0:
            y-=1
            x=x-x.shift(1)
            return differ(x,y)
        else:
            return x
    lags=pd.DataFrame()
    for i in range(0,36):
        lags[i]=x.shift(i)
    fd=pd.DataFrame()
    sd=pd.DataFrame()
    ma=pd.DataFrame()
    for i in range(0,36):
        fd[i]= differ(x,1).shift(i)[0]
        sd[i]= differ(x,2).shift(i)[0]
        ma[i]=x.rolling(3).mean().shift(i)[0]

    msk=x.shift(-1*horiz).isnull().sum(1)+lags.isnull().sum(1)+fd.isnull().sum(1)+ma.isnull().sum(1)+sd.isnull().sum(1)<1
    import numpy as np
    x=np.stack([lags[msk].iloc[:,range(35,-1,-1)],
                     fd[msk].iloc[:,range(35,-1,-1)],
                     sd[msk].iloc[:,range(35,-1,-1)],
                     ma[msk].iloc[:,range(35,-1,-1)]
                    ],axis=2)
    y=ma.shift(-1*horiz)[msk].as_matrix()[:,0]
    return x,y
x,y=make_data(train_data)
x1,y1=make_data(test_data)
pickle.dump([x,y,x1,y1],file=open("ts_example.pck",'wb'))

