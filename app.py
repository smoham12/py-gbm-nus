import numpy as np
import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s:%(name)s:%(message)s', datefmt='%Y-%m-%d,%H:%M:%S'
)

np.random.seed(None)


##############################################################################
st.title('Numerical Demo of Quadratic Variation for Brownian Motion')
##############################################################################


# T: float = 1.0
# N: int = 10
# dt: float = T / N
# mu: float = 0.0
# sigma: float = 1.0

T = st.sidebar.number_input("Time Horizon $(T)$", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
N = st.sidebar.number_input("Number Steps $(N)$", min_value=10, max_value=1000000, value=10, step=10)
mu = st.sidebar.number_input("Drift Rate $(\mu)$", min_value=0.0, value=0.0)
sigma = st.sidebar.number_input("Volatility $(\sigma)$", min_value=0.0, value=1.0)

dt: float = T / N
logger.info(f"The time step dt is {dt}")

#TODO put into config file.
time_col_name = "Time"
gbm_col_name = "BM"
quad_var_col_name = "Quad Var"

dt_array = np.full(N, dt)
dw_array = np.random.normal(mu, sigma*np.sqrt(dt), dt_array.size)
print(dt_array) 
print(dw_array) 

t_array = np.cumsum(np.concatenate((np.zeros(1), dt_array)))
print(t_array) 
w_array = np.cumsum(np.concatenate((np.zeros(1), dw_array)))
print(w_array) 

quad_variation_array = np.cumsum(np.concatenate((np.zeros(1), dw_array*dw_array)))


df = pd.DataFrame({time_col_name: t_array, gbm_col_name: w_array, quad_var_col_name: quad_variation_array})
print(df)

# st.dataframe(df)
st.line_chart(data=df, x=time_col_name, y=[gbm_col_name, quad_var_col_name])

latex_quad_variation_expression = "\lim_{N \\to\infty}\sum_{i=1}^N \\left( W_{i}-W_{i-1} \\right)^2"
st.write(f"The Quadratic Variation of Brownian Motion is: ${latex_quad_variation_expression}$.")

if __name__ == '__main__':
    st.write("hello world from st")