# load packages
from S2S.process import Hindcast, Observations
from S2S.data_handler import ERA5
import pandas as pd

# define input to the hindcast class:
var      = 'sst'

# choose all forecasts of model version CY46R1
t_start  = (2019,7,1)
t_end    = (2020,6,30)

high_res = False
steps    = pd.to_timedelta(range(4,46,7),'D')

bounds = (0,50,50,75)


hindcast = Hindcast(
                    var,
                    t_start,
                    t_end,
                    bounds,
                    high_res=high_res,
                    steps=steps,
                    process=True,
                    download=False,
                    split_work=True
)


# lower-res ERA5 only exists up to 2000-01-01...
t_start_obs = (1999,1,1)
t_end_obs = (2020,9,1)

observations = ERA5(high_res=False).load(
                                        var=var,
                                        start_time=t_start_obs,
                                        end_time=t_end_obs,
                                        bounds=bounds
                                        ) - 273.15


obs_E5 = Observations(name='ERA5',observations=observations,forecast=hindcast)
