import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from scipy import stats
import base64
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.graphics.gofplots import ProbPlot
import scipy.stats as stats
import scipy as sp
import statsmodels.stats.diagnostic as smd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import warnings

warnings.simplefilter("ignore")

def app():

    title_alignment = """ <style>
    .centered-title {
    text-align: center;}
    </style>
    <h1 class="centered-title">Multiple Regression Analysis using ML</h1>
    """
    st.markdown(title_alignment, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;color: blue;'>Edited By Boussiala Mohamed Nachid </h2>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: magenta;'>boussiala.nachid@univ-alger3.dz</h2>", unsafe_allow_html=True)
    
    # Set the title of the app
    
    st.title("Upload CSV or Excel Files or Provide a File Path")

    # File uploader widget for direct uploads
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    # Text input for file path
    file_path = st.text_input("Or enter the file path (local or URL):")

    # Initialize DataFrame
    df = None

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Determine the file type and read the file accordingly
        # Read CSV file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.write("CSV file uploaded successfully!")
        # Read Excel file
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            st.write("Excel file uploaded successfully!")

    # Check if a file path is provided
    if file_path is not None:
        # Check if the file path is a URL or a local file path
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                st.write("CSV Path uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV file from path: {e}")
        elif file_path.endswith('.xlsx'):
            try:
                df = pd.read_excel(file_path)
                st.write("Excel Path uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading Excel file from path: {e}")

    # Display the DataFrame if it has been loaded
    if df is not None:
        st.write("### Preview of uploaded data:")
        st.dataframe(df)

    

        st.title("Treating The Data")
        tab1, tab2, tab3, tab4 = st.tabs(["Raw DataFrame", "Preprocessed DataFrame", "Correlation", "Linear Regression"])

        with tab1:
            st.header("This is Raw of uploading file")
            data = df

            # Display the raw data
            st.write("Display the first few rows of the Data:", data.head())
            st.write("Display the last few rows of the Data: ",data.tail())

            st.write("Generate summary statistics: ",data.describe().T )

        with tab2:


            # Show missing data information
            missing_data= data.isnull().sum()
    
            st.write("Missing Data Information:", missing_data)          

            if missing_data.sum() ==0:
                st.success("No missing values found in the dataset. You can proceed with your analysis!")
            else:

                # Select treatment method
                treatment_method = st.selectbox("Select Treatment Method", 
                                                ["Remove Rows with Missing Values", 
                                                "Fill with Constant", 
                                                "Fill with Mean", 
                                                "Fill with Median", 
                                                "Fill with Mode", 
                                                "Forward Fill", 
                                                "Backward Fill", 
                                                "Interpolate"])

                if st.button("Apply Treatment"):
                    if treatment_method == "Remove Rows with Missing Values":
                        data = data.dropna()
                        st.write("Rows with missing values have been removed.")

                    elif treatment_method == "Fill with Constant":
                        constant_value = st.number_input("Enter constant value to fill:", value=0)
                        data.fillna(constant_value, inplace=True)
                        st.write(f"Missing values filled with {constant_value}.")

                    elif treatment_method == "Fill with Mean":
                        for column in data.select_dtypes(include=['float64', 'int64']).columns:
                            data[column].fillna(data[column].mean(), inplace=True)
                            st.write("Missing values filled with the mean of each column.")

                    elif treatment_method == "Fill with Median":
                        for column in data.select_dtypes(include=['float64', 'int64']).columns:
                            data[column].fillna(data[column].median(), inplace=True)
                            st.write("Missing values filled with the median of each column.")

                    elif treatment_method == "Fill with Mode":
                        for column in data.select_dtypes(include=['object']).columns:
                            data[column].fillna(data[column].mode()[0], inplace=True)
                            st.write("Missing values filled with the mode of each column.")

                    elif treatment_method == "Forward Fill":
                        data.fillna(method='ffill', inplace=True)
                        st.write("Missing values filled using forward fill.")

                    elif treatment_method == "Backward Fill":
                        data.fillna(method='bfill', inplace=True)
                        st.write("Missing values filled using backward fill.")

                    elif treatment_method == "Interpolate":
                        data.interpolate(method='linear', inplace=True)
                        st.write("Missing values filled using interpolation.")

                # Show the cleaned data
                st.write("Cleaned Data", data)   

                # Save cleaned data
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Data as CSV",
                    data=csv,
                    file_name='cleaned_data.csv',
                    mime='text/csv',
                    )

        with tab3:
            st.header('Correlation')

            st.markdown(" <h4 style='text-align: right; color: blue; font-size:18;'> Calculate Pearson Correlation:</h3>", unsafe_allow_html=True)

            # Calculate Corr

            # Select numeric columns
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.warning("Please upload a dataset with at least two numeric columns.")
            else:
                # Calculate the Pearson correlation matrix
                correlation_matrix = df[numeric_columns].corr(method='pearson')

                # Display the correlation matrix
                st.write("Pearson Correlation Matrix:", correlation_matrix)


                # Create a heatmap
                st.markdown(" <h4 style='text-align: right; color: blue; font-size:18;'> display heatmap correlation plot:</h4>", unsafe_allow_html=True)

                #st.write("### display heatmap correlation plot ")
                plt.style.use('Solarize_Light2')
                fig, ax = plt.subplots(figsize=(10, 4), facecolor= 'lightblue')
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f",linewidth=.5, cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linecolor='black')
                ax.set_title (" display heatmap correlation plot ", color='darkorange',
                            font='georgia', 
                            fontweight= 'bold',
                            fontsize= 14)
                ax.grid(True, linestyle='--', alpha=0.7)
                    
                # Display the heatmap in Streamlit
                #st.pyplot(fig)
                st.write(fig)

            st.markdown(" <h4 style='text-align: right; color: blue; font-size:18;'> PairGrid Visualization:</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(15, 10), facecolor= 'lightblue')
            g=sns.PairGrid(data[numeric_columns])
            g.map_diag(sns.histplot)
            g.map_offdiag(sns.scatterplot)
            g.add_legend(True)
            # Display the PairGrid
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle("PairGrid Visualization", color='darkorange',
                            font='georgia', 
                            fontweight= 'bold',
                            fontsize= 35)
            st.pyplot(g.fig)   



        with tab4:
            
            # Select the dependent variable
            target  = st.selectbox("Select the dependent variable:", data.columns)

            # Select independent variables
            features  = st.multiselect("Select independent variables:", data.columns)

            # Input for random_state
            random_state = st.number_input("Enter random state (seed value)", value=42)

            # Input for test_size

            test_size = st.number_input("Enter Test size", value=0.2)
            if test_size <= 0 or test_size >= 1:
                st.error("Test size must be between 0.0 and 1.0.")

        
            if st.button("Run Regression"):
                # Regression Results
                st.subheader("Regression Results")

                if target in features :
                    st.warning("The dependent variable must not be included in the independent variables.")        
            
                elif target and features : 
                    X = data[features]
                    y = data[target]

                                      

                        
                    # Create and train the linear regression model

                    st.markdown(" <h3 style='text-align: right; color: lightblue; font-size:18;'> Run Regression with pure variables:</h3>", unsafe_allow_html=True)

                    # Add a constant to the independent variables
                    X = sm.add_constant(X)

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


                    model= LinearRegression().fit(X_train, y_train)
                    #st.subheader("Model Summary")
                    #st.text(model.summary().as_text())

                    # Download link for regression results
                    #summary_html = model.summary().as_html()
                    #b64 = base64.b64encode(summary_html.encode()).decode()
                    #href = f'<a href="data:text/html;base64,{b64}" download="regression_results.html">Download regression results as HTML</a>'
                    #st.markdown(href, unsafe_allow_html=True)

                    # Display coefficients
                    #results = pd.DataFrame(list(zip(features, model.coef_[1:])),        
                                            #columns=["Variable", "Coefficient"])
                    #st.dataframe(results)

                    #features
                    #model.coef_[1:]
                    #model.intercept_


                    coefficients = pd.DataFrame(model.coef_[1:], features, columns=['Coefficient'])
                    st.write("Model Coefficients:")
                    st.dataframe(coefficients)

                    # Display regression equation
                    st.markdown(" <h4 style='text-align: right; color: green; font-size:18;'> Display regression equation:</h4>", unsafe_allow_html=True)

                    equation = f"{target} = {model.intercept_:.2f}"
    
                    for i in range(0, len(coefficients)):
                        equation += f" + ({coefficients.iloc[i, 0]:.2f} * {features[i]})"

                    st.write("Regression Equation:")
                    st.write(equation)


                    # Predictions
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)



                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred_test)
                    r2 = r2_score(y_test, y_pred_test)
                    mae= mean_absolute_error(y_test, y_pred_test)
                    rmse= root_mean_squared_error(y_test, y_pred_test)

                    # Display results
                    #st.write("Model Performance on Test set:")
                    st.markdown(" <h4 style='text-align: right; color: green; font-size:18;'> Model Performance on Test set: </h4>", unsafe_allow_html=True)

                    output= pd.DataFrame({
                        'Mean Squared Error (MSE)':[round(mse, 2)],
                        "R² Score": [round(r2, 2)],
                        'Mean Absolute Error (MAE)' :[round(mae, 2)],
                        'Root Mean Square Error (RMSE)': [round(rmse, 2)]
                    })
                    st.dataframe(output)
                    #st.write(f"Mean Squared Error: {mse:.2f}")
                    #st.write(f"R² Score: {r2:.2f}")
                    #st.write(f"Mean Absolute Error: {mae:.2f}")


                    # calculate the residuals
                    resid_train = y_train - y_pred_train
                    resid_test = y_test- y_pred_test

                    st.markdown(" <h4 style='text-align: right; color: green; font-size:18;'> Plotting Residuals: </h4>", unsafe_allow_html=True)

                    # Scatter plot the training data
                    plt.style.use('Solarize_Light2')
                    fig, ax = plt.subplots(figsize= (12, 6), facecolor= 'lightblue')
                    train = plt.scatter(x = y_pred_train, y = resid_train , c = 'b', alpha=0.5, marker='D')

                    # Scatter plot the testing data
                    test = plt.scatter(y_pred_test, resid_test , c = 'r', alpha=0.5, marker= '^')

                    # Plot a horizontal axis line at 0
                    ax.hlines(y = 0, xmin = -10, xmax = 110, linewidth= 2, color='k',linestyles = 'dashed' )

                    # Labels
                    ax.legend((train, test), ('Training','Test'), loc='upper left')
                    ax.set_title('Residual Plots', color='darkorange',
                                font='georgia', 
                                fontweight= 'bold',
                                fontsize= 18)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.xticks(rotation= 45)
                    st.write(fig)
                    



            
if __name__ == "__main__":
    app()