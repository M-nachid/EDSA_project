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

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.graphics.gofplots import ProbPlot
import scipy.stats as stats

import warnings

warnings.simplefilter("ignore")



def main():
    title_alignment = """ <style>
    .centered-title {
    text-align: center;}
    </style>
    <h1 class="centered-title">Multiple Regression Analysis with Diagnostics</h1>
    """
    st.markdown(title_alignment, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;color: green;'>Edited By Boussiala Mohamed Nachid </h2>", unsafe_allow_html=True)
    #st.title(""":green[Edited By Boussiala Mohamed Nachid]""")
    st.markdown("<h2 style='text-align: center; color: magenta;'>boussiala.nachid@univ-alger3.dz</h2>", unsafe_allow_html=True)
    #st.title(""":blue[boussiala.nachid@univ-alger3.dz]""")  # Added name
    
    st.markdown(" <h3 style='text-align: center; color: darkgray; font-size:18;'> Multiple Regression is a statistical technique used to model the relationship between "
                 "one dependent variable and two or more independent variables. The goal of multiple regression is to understand how the dependent variable " 
                 "changes when any of the independent variables are varied while keeping the other independent variables constant.</h3>", unsafe_allow_html=True)
    st.write("Key Points:")
    st.write("**Dependent Variable**: The outcome or response variable that you are trying to predict or explain.")

    st.write("**Independent Variables**: The predictors or explanatory variables that are used to predict the dependent variable.")
             
    st.write("**Equation**: The relationship is typically expressed in the form of a linear equation:")

    
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
    #else:
        #if uploaded_file is None and not file_path:
            #st.error("You must either upload a file or enter a valid file path.")
        #else:
            
            #st.info("Please upload a file or enter a valid file path.")
    
        # Display the first few rows of the Data
        st.markdown(" <h4 style='text-align: right; color: wite; font-size:18;'> Display the 6 first Data</h3>", unsafe_allow_html=True)
        st.dataframe(df.head())
        st.markdown(" <h4 style='text-align: right; color: wite; font-size:18;'> Display the last 6 Data</h3>", unsafe_allow_html=True)
        st.dataframe(df.tail())

        st.write("### Generate summary statistics: ")
            # Generate summary statistics
        st.dataframe(df.describe().T)

        st.write("### Calculate Pearson Correlation: ")
            # Calculate Corr

        # Select numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Please upload a dataset with at least two numeric columns.")
        else:
            # Calculate the Pearson correlation matrix
            correlation_matrix = df[numeric_columns].corr(method='pearson')

            # Display the correlation matrix
            st.write("Pearson Correlation Matrix:")
            st.dataframe(correlation_matrix)

            # Create a heatmap
            st.write("### display heatmap correlation plot ")
            plt.style.use('Solarize_Light2')
            fig, ax = plt.subplots(figsize=(10, 4), facecolor= 'lightblue')
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f",linewidth=.5, cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linecolor='black')
            ax.set_title (" display heatmap correlation plot ", color='darkorange',
                          font='georgia', 
                          fontweight= 'bold',
                          fontsize= 18)
            ax.grid(True, linestyle='--', alpha=0.7)
                    
            # Display the heatmap in Streamlit
            #st.pyplot(fig)
            st.write(fig)

#**********************************************************************


        # Select the dependent variable
        dependent_var = st.selectbox("Select the dependent variable:", df.columns)

        # Select independent variables
        independent_vars = st.multiselect("Select independent variables:", df.columns)

        
        if st.button("Run Regression"):
            # Regression Results
            st.subheader("Regression Results")

            if dependent_var in independent_vars:
                st.warning("The dependent variable must not be included in the independent variables.")        
            
            elif dependent_var and independent_vars: 
                X = df[independent_vars]
                y = df[dependent_var]

                # Add a constant to the independent variables
                X = sm.add_constant(X)

                # Fit the regression model
                model = sm.OLS(y, X).fit()

                # Display the regression results
                st.write("Regression Results:")
                st.write(model.summary())

                # Display the regression equation
                st.subheader("Regression Equation")
                equation = f"{dependent_var} = {model.params[0]:.4f}"
                for i in range(1, len(model.params)):
                    equation += f" + {model.params[i]:.4f} * {X.columns[i]}"

                st.write('*'*52)
                st.write(f"{'The Equation of the Regression is : '}  {equation}")
                st.write('*'*52)

                # Extract key statistics

                output= pd.DataFrame({
                     'coefficients' : [model.params],
                     'coefficients' : [model.params],
                     'std_errors' : [model.bse],
                     't_values': [model.tvalues],
                     'p_values' : [model.pvalues],
                     'r_squared': [model.rsquared],
                     'adj_r_squared' : [model.rsquared_adj],
                     'f_stat' : [model.fvalue],
                     'f_pvalue' : [model.f_pvalue]
                     

                })

                #st.dataframe(output)



                # Diagnostic Tests
                st.markdown(" <h2 style='text-align: center; color: blue; font-size:18;'> Diagnostic Tests </h2>", unsafe_allow_html=True)
                
                st.markdown(" <h3 style='text-align: right; color: green; font-size:18;'> Residual Analysis by Plotting </h3>", unsafe_allow_html=True)

                # Extract residuals and fitted values

                residuals= model.resid
                fitted_values= model.fittedvalues
                standardized_residuals= model.get_influence().resid_studentized_internal

                # plot the residuals:

                fig, axes= plt.subplots(2, 2, figsize=(20,15) , facecolor='lightblue')
                axes[0,0].scatter(fitted_values, residuals, marker='D', color='r')
                axes[0,0].axhline(y= 0, color='blue', linewidth=2)
                axes[0,0].set_xlabel('Fitted Value')
                axes[0,0].set_ylabel('Residuals')
                axes[0,0].set_title('Fitted Value vs Residuals', color='darkslateblue',
                                    font='georgia', 
                                    fontweight= 'bold',
                                    fontsize= 14)
                axes[0,0].grid(True, linestyle='--', alpha=0.7)

                # 2. Q-Q Plot
                #QQ = ProbPlot(standardized_residuals)
                #QQ.qqplot(line='45', ax=axes[0, 1], marker='D', color='r')
                #axes[0, 1].set_title('Q-Q Plot of Standardized Residuals', color='darkslateblue',
                                    #font='georgia', 
                                    #fontweight= 'bold',
                                    #fontsize= 14)
                # 2. Normal Q-Q plot
                sm.qqplot(model.resid, line='s', ax=axes[0, 1], marker='D', color='r')
                axes[0,1].set_title('Q-Q Plot of Standardized Residuals', color='darkslateblue',
                                    font='georgia', 
                                    fontweight= 'bold',
                                    fontsize= 14)
                axes[0,0].grid(True, linestyle='--', alpha=0.7)



                # 3. Scale-Location Plot
                axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)),  marker='D', color='r')
                axes[1, 0].set_xlabel('Fitted Values')
                axes[1, 0].set_ylabel('√|Standardized Residuals|')
                axes[1, 0].set_title('Scale-Location Plot',  color='darkslateblue',
                                    font='georgia', 
                                    fontweight= 'bold',
                                    fontsize= 14)
                axes[1, 0].grid(True, linestyle='--', alpha=0.7)

                # 4. Residuals vs Leverage
                leverage = model.get_influence().hat_matrix_diag
                axes[1, 1].scatter(leverage, standardized_residuals , marker='D', color='r')
                axes[1, 1].axhline(y=0, color='b', linestyle='-')
                axes[1, 1].set_xlabel('Leverage')
                axes[1, 1].set_ylabel('Standardized Residuals')
                axes[1, 1].set_title('Residuals vs Leverage', color='darkslateblue',
                                    font='georgia', 
                                    fontweight= 'bold',
                                    fontsize= 14)
                axes[1, 1].grid(True, linestyle='--', alpha=0.7)

                st.write(fig)

                st.markdown(" <h3 style='text-align: right; color: green; font-size:18;'> Normality of Residuals by Plotting /Caculation </h3>", unsafe_allow_html=True)
                
                st.markdown(" <h4 style='text-align: right; color: purple; font-size:18;'> Normality of Residuals by Plotting </h4>", unsafe_allow_html=True)

                # Histogram of residuals
                fig, ax= plt.subplots(figsize=(10, 6),  facecolor='lightblue')
                ax.hist(residuals, bins=10, alpha=0.5, color='blue', edgecolor='red')
                ax.axvline(x=0, color='red', linestyle='--')
                ax.set_title('Histogram of Residuals')
                ax.set_xlabel('Residuals')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.write(fig)
                
                st.markdown(" <h4 style='text-align: right; color: purple; font-size:18;'> Normality of Residuals by Caculation </h4>", unsafe_allow_html=True)

                # Shapiro-Wilk test for normality
                shapiro_test = stats.shapiro(residuals)
                st.markdown(" <h5 style='text-align: left; color: darkviolet; font-size:18;'> Shapiro-Wilk Test for Normality: </h5>", unsafe_allow_html=True)

                SW=pd.DataFrame({
                    "shapiro Wilk Stat" : [shapiro_test[0]],
                    "pValue": [shapiro_test[1]]
                })
                st.dataframe(SW)
                if shapiro_test[1] > 0.05:
                    st.success("Conclusion: Fail to Reject the null hypothesis. Residuals are normally distributed.")
                else:
                    st.warning("Conclusion: Reject the null hypothesis. Residuals are not normally distributed.")

                # Breusch-Pagan test
                st.markdown(" <h5 style='text-align: left; color: darkviolet; font-size:18;'> Breusch-Pagan test for Heteroskedasticity </h5>", unsafe_allow_html=True)

                
                bp_test = het_breuschpagan(residuals, X)
                bp=pd.DataFrame({
                    "LM statistic" : [bp_test[0]],
                    "pValue": [bp_test[1]],
                    "f-value": [bp_test[2]],
                    "f-pValue": [bp_test[3]]
                })
                st.dataframe(bp)
                if bp_test[1] > 0.05:
                    st.success("Conclusion: Fail to Reject the null hypothesis. Residuals are Homoskedastic.")
                else:
                    st.warning("Conclusion: Reject the null hypothesis. Residuals are Heteroskedastic.")

                # White test
                st.markdown(" <h5 style='text-align: left; color: darkviolet; font-size:18;'> White test for Heteroskedasticity </h5>", unsafe_allow_html=True)

                wt_test = het_white(residuals, X)
                bp=pd.DataFrame({
                    "LM statistic" : [wt_test[0]],
                    "pValue": [wt_test[1]]
                })
                st.dataframe(bp)
                if wt_test[1] > 0.05:
                    st.success("Conclusion: Fail to Reject the null hypothesis. Residuals are Homoskedastic.")
                else:
                    st.warning("Conclusion: Reject the null hypothesis. Residuals are Heteroskedastic.")

                st.markdown(" <h4 style='text-align: right; color: purple; font-size:18;'> Multicollinearity </h4>", unsafe_allow_html=True)
                
                
                # Multicollinearity: Variance Inflation Factor (VIF)
                st.markdown(" <h5 style='text-align: left; color: darkviolet; font-size:18;'>  Multicollinearity: Variance Inflation Factor (VIF) </h5>", unsafe_allow_html=True)

                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                st.write("Variance Inflation Factor (VIF):")
                st.dataframe(vif_data)

                # Model Specification
                st.markdown(" <h5 style='text-align: left; color: darkviolet; font-size:18;'> Ramsey RESET Test for Model Specification </h5>", unsafe_allow_html=True)
                try:
                    reset_test = linear_reset(model, power=2)
                    rt=pd.DataFrame({
                        "F-statistic" : [reset_test[0]],
                        "pValue": [reset_test[1]]

                    })
                    st.dataframe(bp)
                    if reset_test[1] > 0.05:
                        st.success("Conclusion: Fail to Reject the null hypothesis. Model is correctly specified")
                    else:
                        st.warning("Conclusion: Reject the null hypothesis. Model is not correctly specified")
                except:
                    st.write("\nRamsey RESET Test: Could not be computed with the current data.")

                # Durbin-Watson Test for Autocorrelation
                st.markdown(" <h5 style='text-align: left; color: darkviolet; font-size:18;'> Durbin-Watson Test for Autocorrelation </h5>", unsafe_allow_html=True)

                dw_statistic = durbin_watson(model.resid)
                st.write(f"Durbin-Watson Statistic: {dw_statistic:.4f}")
                if dw_statistic < 1.5:
                    st.warning("There is evidence of positive autocorrelation.")
                elif dw_statistic > 2.5:
                    st.warning("There is evidence of negative autocorrelation.")
                else:
                    st.success("No autocorrelation detected.")


                
                


if __name__ == "__main__":
    main()  