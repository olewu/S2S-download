#---------try loading the hindcast data with Henrik's routines---------#
from S2S.process import Hindcast
import pandas as pd

var = 'sst'
domain = (5,25,60,75)
t_start = (2020,1,23)
t_end = (2021,1,4)
hres = True
steps = pd.to_timedelta([13,20],'D')

grd_hc = Hindcast(var,t_start,t_end,domain,high_res=hres,steps=steps,process=True,download=False,split_work=True)
