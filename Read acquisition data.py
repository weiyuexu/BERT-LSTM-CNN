import matplotlib.pyplot as plt
from nptdms import TdmsFile
import numpy as np
import pandas as pd

with TdmsFile.open('') as tdms_file:
    for group in tdms_file.groups():      
        group_name = group.name
        print(group_name)
    for channel in group.channels():      
        channel_name = channel.name
        print(channel_name)

    channel = tdms_file['']['']  
    all_channel_data = channel[:]                 
    num = np.array(all_channel_data)-2082844800
    df = pd.DataFrame(num.astype(int))                      
    print(df[:][1:])
    print(df.shape)                          
    channel = np.column_stack((np.array(tdms_file[''][' ']),np.array(tdms_file[''][' '])))
    all_channel_data = channel[:]                 
    all_channel_data[:, 0] = all_channel_data[:, 0] * 1000 + all_channel_data[:, 1]
    num = np.array(all_channel_data)
    df1 = pd.DataFrame(num)                      
    df2 = pd.concat([df, df1], axis=1)
    df2 = df2.iloc[:, 0:2]
    print(df2.shape)    
    df2.to_csv('',index=False)