import numpy as np
import pandas as pd
import streamlit as st
import logging
import altair as alt


def _setup_logging():
    # Check if logging is already configured
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO,  # Set the logging level
                            format='%(asctime)s:%(name)s:%(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

_setup_logging()
logger = logging.getLogger(__name__)
np.random.seed(None)

##############################################################################
st.title('Numerical Demo: Quadratic Variation for Brownian Motion')
##############################################################################

T = st.sidebar.number_input("Time Horizon $(T)$", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
N = st.sidebar.number_input("Number Steps $(N)$", min_value=10, max_value=1000000, value=10, step=10)
mu = st.sidebar.number_input("Drift Rate $(\mu)$", min_value=0.0, value=0.0)
sigma = st.sidebar.number_input("Volatility $(\sigma)$", min_value=0.0, value=1.0)

logger.debug(f'User input are: T={T}, N={N}, mu={mu}, sigma={sigma}')

dt: float = T / N
logger.debug(f"The time step dt is {dt}")

#TODO put into config file.
time_col_name = "Time"
gbm_col_name = "BM"
quad_var_col_name = "Quad Var"

dt_array = np.full(N, dt)
dw_array = np.random.normal(mu, sigma*np.sqrt(dt), dt_array.size)
logger.debug(f'dt_array = \n{dt_array}') 
logger.debug(f'dw_array = \n{dw_array}') 

t_array = np.cumsum(np.concatenate((np.zeros(1), dt_array)))
logger.debug(f't_array = \n{t_array}') 
w_array = np.cumsum(np.concatenate((np.zeros(1), dw_array)))
logger.debug(f'w_array = \n{w_array}') 
quad_variation_array = np.cumsum(np.concatenate((np.zeros(1), dw_array*dw_array)))
logger.debug(f'quad_variation_array = \n{quad_variation_array}') 

df = pd.DataFrame({time_col_name: t_array, gbm_col_name: w_array, quad_var_col_name: quad_variation_array})
logger.debug(f'data = \n{df}')
melted_data = df.melt(id_vars=time_col_name, value_vars=[gbm_col_name, quad_var_col_name], var_name='variable', value_name='Value')
logger.debug(f'melted_data  = \n{melted_data}')

# st.dataframe(df)
# st.line_chart(data=df, x=time_col_name, y=[gbm_col_name, quad_var_col_name])

nb_stdev = 3
y_min, y_max = mu*T - nb_stdev*sigma*np.sqrt(T), mu*T + nb_stdev*sigma*np.sqrt(T)
chart = alt.Chart(melted_data).mark_line().encode(
    x='Time',
    y=alt.Y('Value', scale=alt.Scale(domain=[y_min, y_max])),  # Set y-axis range
    color='variable:N' 
).properties(
    # title='Line Chart with Fixed Y-Axis Range'
)

# Display the chart in Streamlit
st.text("")
st.altair_chart(chart, use_container_width=True)

# Latex Formula
latex_quad_variation_expression = "\lim\limits_{N \\to\infty}\,\,\,\sum\limits_{i=1}^N \\left( W_{i}-W_{i-1} \\right)^2"
st.write(f"The Quadratic Variation of Brownian Motion is :  ${latex_quad_variation_expression}$.")
