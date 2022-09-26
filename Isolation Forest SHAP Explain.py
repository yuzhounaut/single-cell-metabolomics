import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['figure.dpi'] = 300

DATASET_PATH=r'2000cellsPA20220407 all peaks hepg2.csv'
cell_data_df= pd.read_csv(DATASET_PATH,low_memory=False)
cell_data_df.head()
cell_data_df.columns
#specify the 990 features column names to be modelled, dependent on your obtained MS features
to_model_columns=cell_data_df.columns[1:990]

from sklearn.ensemble import IsolationForest
CIF=IsolationForest(n_estimators=100, max_samples='auto', \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
CIF.fit(cell_data_df[to_model_columns])

pred = CIF.predict(cell_data_df[to_model_columns])
cell_data_df['anomaly']=pred
outliers=cell_data_df.loc[cell_data_df['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(cell_data_df['anomaly'].value_counts())
print(outliers)


#Interpretation of Isolation Forest with SHAP (Shapley Additive explanations)
#Compute SHAP values
#https://learn-scikit.oneoffcoder.com/shap.
#https://www.kaggle.com/code/dansbecker/shap-values
# package used to calculate Shap values
import shap
# Create object that can calculate shap values
explainer = shap.Explainer(CIF)
# Calculate Shap values
shap_values = explainer.shap_values(cell_data_df[to_model_columns])
shap_valuesh = explainer(cell_data_df[to_model_columns])
shap_values_df = pd.DataFrame(shap_values)
shap_values_df.to_csv('D:/Download/shap_values_df2.csv', index=False)

shap.initjs()
# Look at the 1st individual and the explanation of values
#It is recommended to inspect abnormal and normal points
shap.force_plot(explainer.expected_value, shap_values[0,:], cell_data_df[to_model_columns].iloc[0,:], 
                plot_cmap="DrDb", matplotlib= True)
# Look at the 12th individual (which is an outlier) and the explanation of values
shap.force_plot(explainer.expected_value, shap_values[12,:], cell_data_df[to_model_columns].iloc[12,:], 
                plot_cmap="DrDb", matplotlib= True)
# Look at the 97th individual and the explanation of values
shap.force_plot(explainer.expected_value, shap_values[97,:], cell_data_df[to_model_columns].iloc[97,:], 
                plot_cmap="DrDb", matplotlib= True)
# Look at the 98th individual (which is an outlier)  and the explanation of values
shap.force_plot(explainer.expected_value, shap_values[98,:], cell_data_df[to_model_columns].iloc[98,:], 
                plot_cmap="DrDb", matplotlib= True)

shap.plots.heatmap(shap_valuesh, max_display=10)

#The Decision Plot for Observation 1
shap.decision_plot(explainer.expected_value, shap_values[0,:], cell_data_df[to_model_columns].iloc[0,:])

shap.summary_plot(shap_values, cell_data_df[to_model_columns])
shap.summary_plot(shap_values, cell_data_df[to_model_columns], plot_type="bar")
# visualize the first 5 predictions explanations with a dark red dark blue color map.
shap.force_plot(explainer.expected_value, shap_values[0:5,:], cell_data_df[to_model_columns].iloc[0:5,:],
                plot_cmap="DrDb", show=False)



