import streamlit as st
import pandas as pd
from pandas import CategoricalDtype
from lifelines.datasets import load_rossi
from lifelines import WeibullAFTFitter
from utils import plotter, read_config
from joblib import dump, load
import json



# SETUP
# st.set_page_config(layout="wide")
df = pd.read_csv('./data/imputed_consumer.csv', nrows=100)
model = load('model/churn.joblib')


with open('scoring_dict.json', 'r') as f:
    scoring_dict = json.loads(f.read())


DURATION = 'duration'
EVENT = 'observed'
cfg = read_config('data_dictionary.yaml')


# INDIVIDAL PREDICTIONS 
st.sidebar.title("Individual Prediction")
slider_1 = st.sidebar.slider('bd', 0, 100)
slider_2 = st.sidebar.slider('latest_payment_plan_days',0, 30)
slider_3 = st.sidebar.slider('avg_num_unq',0, 100)
slider_4 = st.sidebar.slider('duration', 0, 50)
slider_5 = st.sidebar.selectbox('is_auto_renew_1', [0, 1])
slider_6 = st.sidebar.selectbox('observed', [0, 1])
slider_7 = st.sidebar.selectbox('gender_1.0', [0, 1])

select_options = [x for x in scoring_dict.keys() if 'latest_payment_method' in x]
selectbox = st.sidebar.selectbox('Select ONE', select_options)

scoring_dict['bd'] = slider_1
scoring_dict['latest_payment_plan_days'] = slider_2
scoring_dict['avg_num_unq'] = slider_3
scoring_dict['duration'] = slider_4
scoring_dict['is_auto_renew_1'] = slider_5
scoring_dict['observed'] = slider_6
scoring_dict['gender_1.0'] = slider_7
scoring_dict[selectbox] = 1


prediction_output = model.predict_expectation(pd.DataFrame(scoring_dict),conditional_after=pd.DataFrame(scoring_dict)['duration']).values[0].round(0).astype(int)
#predict_input = pd.DataFrame([week, 0, fin, age, 1, 1, mar, paro, 1]).T
#predict_input.columns = ['week', 'arrest', 'fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
st.sidebar.write("## Weeks until churn:", round(prediction_output))

# custom features

# STREAMLIT CODE
st.title('KKBox Survival Analysis')
st.write("Data source: " + 'https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data')
st.write('''In this challenge, you are asked to predict whether a user will churn after his/her subscription expires. Specifically, we want to forecast if a user make a new service subscription transaction within 30 days after the current membership expiration date.''')

col1 = st.beta_columns(1)

# KAPLAN MEIER CURVES
drop_cols = [
        'customer_id',
        'bd',
        'city',
        'reg_month',
        'tx_last_date',
        'mem_end_date',
        'latest_actual_amount_paid',
        'latest_payment_method_id',
        'avg_tot_secs',
        'avg_num_unq',
        'duration'
        ]
st.title('Kaplan-Meier Curves')
option = st.selectbox(
        '',
     [x for x in df.columns if x not in drop_cols])



plt = plotter(df, option, DURATION, EVENT, CategoricalDtype)
KM_plot = st.pyplot(plt)


st.title("Model Summary")
st.write(model.summary)

# COX PROPORTIONAL HAZARDS SUMMARY
#from lifelines import CoxPHFitter
#rossi= load_rossi()

#st.title('Regression Model Summary')
#cph = CoxPHFitter()
#cph.fit(df, duration_col=DURATION, event_col=EVENT)

#st.write("## Coefficients")
#cols = ['coef','exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
#cph.summary[cols]

#st.write("## Summary")
#col_2 = [
#    cph._n_examples,
#    sum(cph.event_observed),
#    cph.baseline_estimation_method,
#    cph.event_col,
#    cph.duration_col,
#    cph._class_name,
#    cph.log_likelihood_,
#    cph.concordance_index_,
#    cph.AIC_partial_]

#col_1 = [
#    'observations',
#    'events_oberved',
#    'baseline estimation',
#    'event column',
#    'duration column',
#    'model',
#    'log likelihood',
#    'concordance',
#    'partial AIC'    
#]
#results = pd.DataFrame([col_1,col_2]).T
#results.columns = ['', ' ']
#results.set_index('')
#results



