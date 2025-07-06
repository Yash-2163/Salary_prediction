import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

df=pd.read_csv(r'..\data\adjusted_salary_model_data.csv')

salary_cols_to_drop = ['base_salary', 'bonus', 'stock_options', 'total_salary', 'salary_in_usd', 'adjusted_total_usd']
df_features = df.drop(columns=salary_cols_to_drop)
# Drop 'experience_level' to avoid redundancy with 'experience_level_encoded'
df_features = df_features.drop(columns=['experience_level'])


# # Assuming df is original dataframe and df_features is features dataframe without salary columns
# Keep target separate
y = df['salary_in_usd']
X = df_features

# # For numeric features, get correlation with target
# numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
# correlations = X[numeric_cols].corrwith(y).sort_values(key=abs, ascending=False)
# print(correlations)

# # Optional: Plot heatmap for better visualization
# sns.barplot(x=correlations.values, y=correlations.index)
# plt.title('Correlation of Numeric Features with Total Salary')
# plt.show()

# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# rf.fit(X, y)

# importances = rf.feature_importances_
# feature_importances = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)

# for feature, importance in feature_importances[:10]:  # Top 10
#     print(f"{feature}: {importance:.4f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Initialize RFE with desired number of features, e.g., 10
rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

# Selected features
selected_features = X_train.columns[rfe.support_]
print("Top 10 selected features by RFE:")
print(selected_features)


# print(df.info())