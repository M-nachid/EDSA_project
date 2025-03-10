# Import packges

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, kpss, range_unit_root_test
from statsmodels.tsa.arima_process import arma_generate_sample
import plotly.graph_objects as go
import plotly.express as px

from arch.unitroot import ADF
from arch.unitroot import DFGLS, PhillipsPerron
from statsmodels.tsa.stattools import zivot_andrews

import warnings

warnings.simplefilter("ignore")


def main():
    title_alignment = """ <style>
    .centered-title {
    text-align: center;}
    </style>
    <h1 class="centered-title">Time Series Unit Root Tests</h1>
    """
    st.markdown(title_alignment, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;color: green;'>Edited By Boussiala Mohamed Nachid </h2>", unsafe_allow_html=True)
    #st.title(""":green[Edited By Boussiala Mohamed Nachid]""")
    st.markdown("<h2 style='text-align: center; color: magenta;'>boussiala.nachid@univ-alger3.dz</h2>", unsafe_allow_html=True)
    #st.title(""":blue[boussiala.nachid@univ-alger3.dz]""")  # Added name
    
    st.markdown(" <h3 style='text-align: center; color: purple; font-size:18;'> This app performs various unit root tests such as ADF, Phillips Perron, Kpss and Zivot Andrews on your time series data.\n"
                "Upload an Excel or csv file, select a variable, and choose which tests to run.</h3>", unsafe_allow_html=True)
    
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
        #else:
            #st.warning("Please enter a valid CSV or Excel file path.")

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
            fig, ax = plt.subplots()
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f",linewidth=.5, cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linecolor='black')
            plt.title(" display heatmap correlation plot ")
                    
            # Display the heatmap in Streamlit
            #st.pyplot(fig)
            st.write(fig)



    ########################################################################

        columns= df.columns.tolist()

        st.write('### select 2 columns for correlation:')

        select_col= st.multiselect("", columns, max_selections= 2)

            # Calculate and display correlation and p-value
        
        if len(select_col) == 2:
            col1, col2 = select_col
                # Calculate correlation and p-value
            correlation, p_value = pearsonr(df[col1], df[col2])
            
                # Display results
            st.write(f"Correlation between **{col1}** and **{col2}**: {correlation:.4f}")
            st.write(f"P-value: {p_value:.4f}")

                # Interpret results
  
            if p_value < 0.05:
                st.success(f"Conclusion: The correlation between **{col1}** and **{col2}** is significatif")
            else:
                st.warning(f"Conclusion: The correlation between **{col1}** and **{col2}** is significatif")
        else:
                #st.write("Please select exactly two columns to calculate correlation.")
                st.markdown(" <h4 style='text-align: right; color: red; font-size:18;'> Please select exactly two columns to calculate correlation </h3>", unsafe_allow_html=True)

        ####################################################################################

        # Select columns for plotting
        st.write("## Select 2 columns to plot:")
        x_axis = st.selectbox("Select X-axis column", columns)
        y_axis = st.selectbox("Select Y-axis column", columns)
        
        # Plotting the data
        plt.style.use('Solarize_Light2')
        plt.figure(figsize=(10, 5), facecolor='lightblue')
        plt.plot(df[x_axis], df[y_axis], marker='o', linewidth= 2, color='b')
        plt.title(f"Plotting {y_axis} vs {x_axis}",
                  font= 'georgia',
                  fontsize= 20,
                  fontweight= 'bold',
                  color= 'brown')

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.grid()
        st.pyplot(plt)    

        ########################################################################

                ####################################################################

        st.write('### Select columns to plotting')
        
        columns= df.columns.tolist()
        
        selected_columns= st.multiselect('', columns)
        
        if selected_columns:
            st.write('you selected:', selected_columns)
            st.dataframe(df[selected_columns])
        #else:
            
            #st.write('please select at least one column')
            #st.markdown(" <h4 style='text-align: right; color: red; font-size:18;'> Please select at least one column </h3>", unsafe_allow_html=True)

        
        if selected_columns:
            st.write('you selected:', selected_columns)
            fig, ax = plt.subplots(figsize=(10, 5),  facecolor='lightblue')
            
            for column in selected_columns:
                ax.plot(df[column], label= column, linewidth=2)
            
            plt.title('Plotting differents Variables',
                      font= 'georgia',
                      fontsize= 20,
                      fontweight= 'bold',
                      color= 'brown')
            plt.xlabel('Index')
            plt.ylabel("values")
            plt.xticks(rotation= 45)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.markdown(" <h4 style='text-align: right; color: red; font-size:18;'> Please select at least one variable </h3>", unsafe_allow_html=True)
            #st.write('please select variables')

        ########################################################################
              
        # Select columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            st.error("No numeric columns found in the uploaded file.")
            return
        
        selected_column = st.selectbox("", numeric_columns)

        
        # Get the data
        data = df[selected_column].dropna()

        # Plot the time series
        st.write("### Plotting Time Series before running Unit Root Test:")
        plt.style.use('Solarize_Light2')
        fig, ax = plt.subplots(figsize=(10, 4),   facecolor='lightblue')
        ax.plot(data.values, color= 'b', linewidth=2)
        ax.set_title(f'Plotting {selected_column} variable',
                     font= 'georgia',
                     fontsize= 20,
                     fontweight= 'bold',
                     color= 'brown')

        plt.xticks(rotation=45)
        ax.grid(True)
        st.pyplot(fig)

        ####################################################################
        #st.write("### Select a variable for unit root testing:")
        # Options for tests
        st.write(f"{'#### Apply The Unit Root Tests to : '}{selected_column}")
        run_adf = st.checkbox("Augmented Dickey-Fuller (ADF) Test", value=True)
        run_kpss = st.checkbox("Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test", value=True)
        run_pp = st.checkbox("Phillips-Perron (PP) Test", value=True)
        run_za = st.checkbox("Zivot Andrews (ZA) Test", value=True)
        run_dfgls = st.checkbox("DFGLS Test", value=True)
        run_rur= st.checkbox('Range Unit Root', value=True)
        

        if st.button("Run Tests"):
            st.write("## Test Results")

                        # ADF Test
            if run_adf:
                st.markdown(" <h2 style='text-align: right; color: orange; font-size:18;'> Augmented Dickey-Fuller (ADF) Test </h2>", unsafe_allow_html=True)
                #st.write("# Augmented Dickey-Fuller Test")
                st.write(""" The Augmented Dickey-Fuller tests for a unit root in a univariate process in the presence of serial correlation. 
                         The ADF test handles more complex models and is the typical go-to for most analysts.""")
                st.write (""" **Null Hypothesis**: The series has a unit root (non-stationary)  
                        **Alternative Hypothesis**: The series has no unit root (stationary) """)

                # Perform ADF test with 3 different model specifications
                models = ['n', 'c', 'ct']
                model_descriptions = ['Model 1: No Constant, No Trend', 
                                      'Model 2: Constant, No Trend', 
                                      'Model 3: Constant and Trend']
                #print("Augmented Dickey-Fuller Test Results:".center(50))
                #print("-" * 50)

                #st.write("## ADF Test: Model 1: No Constant, No Trend:")
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> ADF Test: Model 1: No Constant, No Trend: </h4>", unsafe_allow_html=True)

                result = adfuller(data.values, regression='n', autolag='AIC')
                adf_output = pd.DataFrame({
                    'ADF Statistic:' : [result[0]],
                    'p-value:' : [result[1]],
                    'The number of lags used:' : [result[2]],
                    'The number of observations:' : [result[3]]
                })
                
                
                for key, value in result[4].items():
                    adf_output['Critical value (%s)' %key] = value
                print('-'*80)
                # Interpret results

                st.dataframe(adf_output)
                if result[1] < 0.05:
                    st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                print('-'*80)

                ##############################################################################
                #st.write("## ADF Test: Model 2: Constant, No Trend")
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> ADF Test: Model 2: Constant, No Trend </h4>", unsafe_allow_html=True)

                result = adfuller(data.values, regression='c', autolag='AIC')
                adf_output = pd.DataFrame({
                    'ADF Statistic:' : [result[0]],
                    'p-value:' : [result[1]],
                    'The number of lags used:' : [result[2]],
                    'The number of observations:' : [result[3]]
                })
                
                print("Critical Values:".center(35))
                for key, value in result[4].items():
                    adf_output['Critical value (%s)' %key] = value
                print('-'*80)
                # Interpret results

                st.dataframe(adf_output)
                if result[1] < 0.05:
                    st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                #st.write('-'*80)

               ############################################################################## 
                #st.write("## ADF Test: Model 3: Constant and Trend")
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> ADF Test: Model 3: Constant and Trend </h4>", unsafe_allow_html=True)

                result = adfuller(data.values, regression='ct', autolag='AIC')
                adf_output = pd.DataFrame({
                    'ADF Statistic:' : [result[0]],
                    'p-value:' : [result[1]],
                    'The number of lags used:' : [result[2]],
                    'The number of observations:' : [result[3]]
                })
                
                print("Critical Values:".center(35))
                for key, value in result[4].items():
                    adf_output['Critical value (%s)' %key] = value
                print('-'*80)
                # Interpret results
                st.dataframe(adf_output)
                st.write('-'*80)
                if result[1] < 0.05:
                    st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                #st.write('-'*80)
###################################################################################
            # KPSS Test
            if run_kpss:
                #st.write("# Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test")
                st.markdown(" <h2 style='text-align: right; color: orange; font-size:18;'> Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test </h2>", unsafe_allow_html=True)

                st.write(""" KPSS is another test for checking the stationarity of a time series
                **Null Hypothesis**: The series has no unit root (stationary)  
                **Alternative Hypothesis**: The series has a unit root (non-stationary)
                """)

                # Perform kpss test with 2 different model specifications
                models = ['c', 'ct']
                model_descriptions = ['Model 1: Constant', 
                                      'Model 2: Constant and Trend']
                #print("kpss Test Results:".center(50))
                #print("-" * 50)

                #st.write("## KPSS Test: Model 1:  Constant")
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> KPSS Test: Model 1:  Constant </h4>", unsafe_allow_html=True)

                result = kpss(data.values, regression='c')
                kpss_output = pd.DataFrame({
                    'kpss Statistic:' : [result[0]],
                    'p-value:' : [result[1]],
                    'The number of lags used:' : [result[2]]
                })
                
                
                for key, value in result[3].items():
                    kpss_output['Critical value (%s)' %key] = value
                #print('-'*80)
                # Interpret results

                st.dataframe(kpss_output)
                st.write('-'*80)
                if result[1] > 0.05:
                    st.success("Conclusion: Fail to reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Reject the null hypothesis. Series is non-stationary.")
                #st.write('-'*80)
                ############################################################################## 

                #st.write("## KPSS Test: Model 2: Constant and Trend")
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> KPSS Test: Model 2: Constant and Trend </h4>", unsafe_allow_html=True)

                result = kpss(data.values, regression='ct')
                kpss_output = pd.DataFrame({
                    'kpss Statistic:' : [result[0]],
                    'p-value:' : [result[1]],
                    'The number of lags used:' : [result[2]]
                })
                
                
                for key, value in result[3].items():
                    kpss_output['Critical value (%s)' %key] = value
                #print('-'*80)
                # Interpret results

                st.dataframe(kpss_output)
                #st.write('-'*80)
                if result[1] > 0.05:
                    st.success("Conclusion: Fail to reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Reject the null hypothesis. Series is non-stationary.")
                #st.write('-'*80)
            
#######################################################################

                # Phillips-Perron Test (using arch library)
            if run_pp:
                #st.write("#  The Phillips-Perron (PP) Test")
                st.markdown(" <h2 style='text-align: right; color: orange; font-size:18;'> The Phillips-Perron (PP) Test </h2>", unsafe_allow_html=True)

                st.write(""" The Phillips-Perron test is similar to the ADF except that the regression run 
                         does not include lagged values of the first differences. Instead, the PP test fixed the 
                         t-statistic using a long run variance estimation, implemented using a Newey-West covariance estimator.
                         **Null Hypothesis**: The series has a unit root (non-stationary)  
                         **Alternative Hypothesis**: The series has no unit root (stationary)
                         """)
                
                # Perform PP test with 3 different model specifications

                models = ['n', 'c', 'ct']
                model_descriptions = ['Model 1: No trend components', 
                                      'Model 2: Include a constant', 
                                      'Model 3: Include a constant and linear time trend']
                try: 
                    #st.write("### Phillips-Perron Test: Model 1: No trend components")
                    st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Phillips-Perron Test: Model 1: No trend components </h4>", unsafe_allow_html=True)

                    pp = PhillipsPerron(data.values, trend='n')
                    pp_result = pp.summary()
                    pp_output = pd.DataFrame({
                        'Phillips-Perron Statistic:' : [pp.stat],
                        'p-value:' : [pp.pvalue],
                        'The number of lags used:' : [pp.lags]
                        })
                
                    # Add critical values
                    critical_values = pp.critical_values
                    for key in critical_values.keys():
                        pp_output[f'Critical Value ({key})'] = critical_values[key]

              
                    # Interpret results

                    st.dataframe(pp_output)
                    #st.write('-'*80)
                    if pp.pvalue < 0.05:
                        st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                    else:
                        st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                    #st.write('-'*80)

                    st.write("### The Detail of PP test results")  

                    # Detailed PP test results
                    st.markdown(" <h5 style='text-align: right; color: green; font-size:18;'> The Detail of PP test results </h5>", unsafe_allow_html=True)
                    st.text(str(pp_result))

                except Exception as e:
                    st.error(f"Error in Phillips-Perron test: {str(e)}")
                    st.info("Note: This test requires the 'arch' library. Install it using pip install arch")


                #####################################################

                                #####################################################
                
                try: 
                    #st.write("## Phillips-Perron Test: Model 2: Include a constant ")
                    st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Phillips-Perron Test: Model 2: Include a constant </h4>", unsafe_allow_html=True)

                    pp = PhillipsPerron(data.values, trend='c')
                    pp_result = pp.summary()
                    pp_output = pd.DataFrame({
                        'Phillips-Perron Statistic:' : [pp.stat],
                        'p-value:' : [pp.pvalue],
                        'The number of lags used:' : [pp.lags]
                        })
                
                    # Add critical values
                    critical_values = pp.critical_values
                    for key in critical_values.keys():
                        pp_output[f'Critical Value ({key})'] = critical_values[key]
            
                    # Interpret results

                    st.dataframe(pp_output)
                    #st.write('-'*80)
                    if pp.pvalue < 0.05:
                        st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                    else:
                        st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                    #st.write('-'*80)

                    st.markdown(" <h5 style='text-align: right; color: green; font-size:18;'> The Detail of PP test results </h5>", unsafe_allow_html=True)
                    # Detailed PP test results
                    st.text("Detailed Phillips-Perron Test Results:")
                    st.text(str(pp_result))

                except Exception as e:
                    st.error(f"Error in Phillips-Perron test: {str(e)}")
                    st.info("Note: This test requires the 'arch' library. Install it using pip install arch")


                #####################################################
                
                try: 
                    #st.write("## Phillips-Perron Test: Model 3: Include a constant and linear time trend")
                    st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Phillips-Perron Test: Model 3: Include a constant and linear time trend </h4>", unsafe_allow_html=True)

                    pp = PhillipsPerron(data.values, trend='ct')
                    pp_result = pp.summary()
                    pp_output = pd.DataFrame({
                        'Phillips-Perron Statistic:' : [pp.stat],
                        'p-value:' : [pp.pvalue],
                        'The number of lags used:' : [pp.lags]
                        })
                
                    # Add critical values
                    critical_values = pp.critical_values
                    for key in critical_values.keys():
                        pp_output[f'Critical Value ({key})'] = critical_values[key]
            
                    # Interpret results

                    st.dataframe(pp_output)
                    #st.write('-'*80)
                    if pp.pvalue < 0.05:
                        st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                    else:
                        st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                    #st.write('-'*80)
                    
                    
                    st.markdown(" <h5 style='text-align: right; color: green; font-size:18;'> The Detail of PP test results </h5>", unsafe_allow_html=True)

                    # Detailed PP test results
                    st.text("Detailed Phillips-Perron Test Results:")
                    st.text(str(pp_result))

                except Exception as e:
                    st.error(f"Error in Phillips-Perron test: {str(e)}")
                    st.info("Note: This test requires the 'arch' library. Install it using pip install arch")

########################################################################################################################
            # ZA Test 
            models = ['c', 't', 'ct']
            model_descriptions = ['Model 1: constant only', 
                                  'Model 2: trend only', 
                                  'Model 3: constant and trend']        
            if run_za: 
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Zivot-Andrews Test: Model 1: constant only </h4>", unsafe_allow_html=True)

                try:
                    # Perform the Zivot-Andrews test
                    result = zivot_andrews(data.values, regression='c')

                    # Display the results
                    za_output=pd.DataFrame({"Zivot-Andrews Statistic:": [result[0]],
                                            'p-value:' : [result[1]],
                                            'The number of lags used:' : [result[3]],
                                            'The  break period:' : [result[4]]
                    })

                    for key, value in result[2].items():
                        za_output['Critical value (%s)' %key] = value
                
                    # Interpret results

                    st.dataframe(za_output)
                    #st.write('-'*80)
                    if result[1] < 0.05:
                        st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                    else:
                        st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                    #st.write('-'*80)
                except Exception as e:
                    st.error(f"Error in ZA test: {str(e)}")
                    
                #########################################################################

                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Zivot-Andrews Test: Model 2: trend only </h4>", unsafe_allow_html=True)

                try:
                    # Perform the Zivot-Andrews test
                    result = zivot_andrews(data.values, regression='t')

                    # Display the results
                    za_output=pd.DataFrame({"Zivot-Andrews Statistic:": [result[0]],
                                            'p-value:' : [result[1]],
                                            'The number of lags used:' : [result[3]],
                                            'The  break period:' : [result[4]]
                    })

                    for key, value in result[2].items():
                        za_output['Critical value (%s)' %key] = value
                
                    # Interpret results

                    st.dataframe(za_output)
                    #st.write('-'*80)
                    if result[1] < 0.05:
                        st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                    else:
                        st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                    #st.write('-'*80)
                except Exception as e:
                    st.error(f"Error in ZA test: {str(e)}")

                #########################################################################

                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Zivot-Andrews Test: Model 3: constant and trend  </h4>", unsafe_allow_html=True)

                try:
                    # Perform the Zivot-Andrews test
                    result = zivot_andrews(data.values, regression='ct')

                    # Display the results
                    za_output=pd.DataFrame({"Zivot-Andrews Statistic:": [result[0]],
                                            'p-value:' : [result[1]],
                                            'The number of lags used:' : [result[3]],
                                            'The  break period:' : [result[4]]
                    })

                    for key, value in result[2].items():
                        za_output['Critical value (%s)' %key] = value
                
                    # Interpret results

                    st.dataframe(za_output)
                    #st.write('-'*80)
                    if result[1] < 0.05:
                        st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                    else:
                        st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
                    #st.write('-'*80)
                except Exception as e:
                    st.error(f"Error in ZA test: {str(e)}")

########################################################################################

            if run_dfgls:
                
                st.write("""The Dickey-Fuller GLS test is an improved version of the ADF which uses a GLS-detrending regression 
                         before running an ADF regression with no additional deterministic terms. This test is only available 
                         with a constant or constant and time trend (trend='c' or trend='ct')
                         """)
                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Dickey-Fuller GLS (DFGLS) Test: Model 1: constant  </h4>", unsafe_allow_html=True)

                dfgls = DFGLS(data, trend='c')
                # Display the results
                dfgls_output= pd.DataFrame({'Test Statistic:':[dfgls.stat],
                                           'p-value: ' : [dfgls.pvalue],
                                           'The number of lags used:': [dfgls.lags]
                })
                
                
                for key in dfgls.critical_values.keys():
                    dfgls_output[f'Critical Value ({key})'] = dfgls.critical_values[key]
            
                
                st.dataframe(dfgls_output)
                
                if dfgls.pvalue < 0.05:
                    st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")

                st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Dickey-Fuller GLS (DFGLS) Test: Model 2 : Constant and Trend  </h4>", unsafe_allow_html=True)

                dfgls = DFGLS(data, trend='ct')
                # Display the results
                dfgls_output= pd.DataFrame({'Test Statistic:':[dfgls.stat],
                                           'p-value: ' : [dfgls.pvalue],
                                           'The number of lags used:': [dfgls.lags]
                })
                
                
                for key in dfgls.critical_values.keys():
                    dfgls_output[f'Critical Value ({key})'] = dfgls.critical_values[key]
            
                
                st.dataframe(dfgls_output)
                
                if dfgls.pvalue < 0.05:
                    st.success("Conclusion: Reject the null hypothesis. Series is stationary.")
                else:
                    st.warning("Conclusion: Fail to reject the null hypothesis. Series is non-stationary.")
##################################################################################################################

            # Range Unit Root Test
            st.markdown(" <h4 style='text-align: center; color: brown; font-size:18;'> Range Unit Root Test (RUR): Model 2 : Constant and Trend  </h4>", unsafe_allow_html=True)

            if run_rur:
                #st.write("### Range Unit Root Test")
                st.write('''the RUR test outperforms the power of standard unit-root tests on near-unit-root stationary time series; 
                         it is invariant with respect to the innovations distribution and asymptotically immune to noise. An extension of the RUR test, 
                         called the forward–backward range unit-root (FB-RUR) improves the check in the presence of additive outliers''')
                st.write("""
                **Null Hypothesis**: The series has a unit root (non-stationary)  
                **Alternative Hypothesis**: The series has no unit root (stationary)
                """)  # Corrected Hypotheses

                try:
                    result = range_unit_root_test(data.values, store=True)

                    rur_output = pd.DataFrame({
                        'Statistic': [result[0]],
                        'p-value': [result[1]],
                    })

                    # Add critical values
                    for key, value in result[2].items():
                        rur_output[f'Critical Value ({key})'] = value

                    st.dataframe(rur_output)

                    if result[1] <= 0.05:
                        st.success(" Reject the null hypothesis, The series is stationary")  # Corrected Interpretation
                    else:
                        st.warning(
                            " Fail to reject the null hypothesis, The series is non-stationary")  # Corrected Interpretation
                        

                except Exception as e:
                    st.error(f"Error in Range Unit Root test: {str(e)}")

                
                

           


    

if __name__ == "__main__":
    main()   
