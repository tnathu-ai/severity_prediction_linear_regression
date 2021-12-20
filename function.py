# import libraries general libraries
import pandas as pd
import numpy as np

# Modules for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

plt.rcParams['figure.figsize'] = [6, 6]

# ignore DeprecationWarning Error Messages
import warnings

warnings.filterwarnings('ignore')


def style(table):
    """
    quick styling
    style: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html
    color: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    view = table.style.background_gradient(cmap='Pastel1')
    return view


def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple. 
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)[
        round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# Describing data
def group_median_aggregation(df, group_var, agg_var):
    # Grouping the data and taking median
    grouped_df = df.groupby([group_var])[agg_var].median().sort_values(ascending=False)
    return grouped_df


# function that imputes median
def impute_median(series):
    return series.fillna(series.median())


def calculate_min_max_whisker(df):
    """
    Calculates the values of the 25th and 75th percentiles
    It takes the difference between the two to get the interquartile range (IQR).
    Get the length of the whiskers by multiplying the IQR by 1.5
    Calculate the min and max whisker value by subtracting
    Add the whisker length from the 25th and 75th percentile values.
    """
    q_df = df.quantile([.25, .75])
    q_df.loc['iqr'] = q_df.loc[0.75] - q_df.loc[0.25]
    q_df.loc['whisker_length'] = 1.5 * q_df.loc['iqr']
    q_df.loc['max_whisker'] = q_df.loc['whisker_length'] + q_df.loc[0.75]
    q_df.loc['min_whisker'] = q_df.loc[0.25] - q_df.loc['whisker_length']
    return q_df


def whitespace_remover(df):
    """
    The function will remove extra leading and trailing whitespace from the data.
    Takes the data frame as a parameter and checks the data type of each column.
    If the column's datatype is 'Object.', apply strip function; else, it does nothing.
    Use the whitespace_remover() process on the data frame, which successfully removes the extra whitespace from the columns.
    https://www.geeksforgeeks.org/pandas-strip-whitespace-from-entire-dataframe/
    """
    # iterating over the columns
    for i in df.columns:

        # checking datatype of each columns
        if df[i].dtype == 'str':

            # applying strip function on column
            df[i] = df[i].map(str.strip)
        else:
            # if condition is False then it will do nothing.
            pass


"""
 Impute missing values by taking category-specific numerical and categorical imputations
 Credit: https://towardsdatascience.com/pandas-tricks-for-imputing-missing-data-63da3d14c0d6
 """


def impute_numerical(df, categorical_column, numerical_column):
    frames = []
    # within a for-loop we can define column-specific data frames:
    for i in list(set(df[categorical_column])):
        df_category = df[df[categorical_column] == i]
        # we can fill the missing values in these column-specific data frames with their respective median of numerical column:
        if len(df_category) > 1:
            # checking the length of the data frame within the for loop
            # imputing with the column-specific median if the length is greater than one
            df_category[numerical_column].fillna(df_category[numerical_column].median(), inplace=True)
        else:
            # If the length is equal to 1 we impute with the median across all countries
            df_category[numerical_column].fillna(df[numerical_column].median(), inplace=True)
        # We then append the result to a list we’ll call “frames”
        frames.append(df_category)
        final_df = pd.concat(frames)
    return final_df


def impute_categorical(df, categorical_column1, categorical_column2):
    cat_frames = []
    for i in list(set(df[categorical_column1])):
        df_category = df[df[categorical_column1] == i]
        if len(df_category) > 1:
            df_category[categorical_column2].fillna(df_category[categorical_column2].mode()[0], inplace=True)
        else:
            df_category[categorical_column2].fillna(df[categorical_column2].mode()[0], inplace=True)
        cat_frames.append(df_category)
        # concatenate the resulting list of data frames:
        cat_df = pd.concat(cat_frames)
    return cat_df


# # generate box plots for any numerical column
# def boxplot(column):
#     sns.boxplot(data=df,x=df[f”{column}”])
#     plt.title(f”Boxplot of {df} {column}”)
#     plt.show()


def convert_categories(cat_list):
    for col in cat_list:
        df[col] = df[col].astype('category')
        df[f'{col}_cat'] = df[f'{col}_cat'].cat.codes


##########################################################################################################################

def plot_model_feature_importances(model):
    '''
    Custom function to plot the 
    feature importances of the classifier.
    '''
    fig = plt.figure()

    # get the feature importance of the classifier 'model'
    feature_importances = pd.Series(model.feature_importances_,
                                    index=X_train.columns) \
        .sort_values(ascending=False)

    # plot the bar chart
    sns.barplot(x=feature_importances, y=X_train.columns)
    plt.title('Classifier Feature Importance', fontdict={'fontsize': 20})
    plt.xticks(rotation=60)
    plt.show()


def model(X_train, X_test, y_train, y_test):
    # statsmodels
    features = X_train.copy()
    features['mpg'] = y_train

    formula = 'mpg~' + '+'.join(X_train.columns)
    model = ols(formula=formula, data=features).fit()

    # sklearn
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_hat_train = linreg.predict(X_train)
    y_hat_test = linreg.predict(X_test)

    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)

    print("Train R2: ", linreg.score(X_train, y_train))
    print("Test R2: ", linreg.score(X_test, y_test))

    print("Train RMSE: ", train_mse ** 0.5)
    print("Test RMSE: ", test_mse ** 0.5)

    return model.summary()
