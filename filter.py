
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import math 
import logging  

##############################################################################

# Function to apply Hamilton filter:
def hamiltonFilter(array, p: int = 2, h: int = 4):
    """ Takes an array and returns two arrays - the predicted evolution of
    the series ("trend"), and the deviation from that prediction ("cycle").
    """
    
    ind = array.index
    nm = array.name
    array = np.asarray(array)
    y = pd.DataFrame(array, columns = ['Y'])
    y.dropna(inplace=True)    
    
    # Creating X matrix - break off as separate function
    x = y[['Y']].copy()
    x.rename(columns = {'Y':'lagP'}, inplace=True)
    x['lagP'] = x['lagP'].shift(h)
    
    i = 0
    while i < (p-1):
        name = 'lagP' + str(1+i)
        x[name] = x['lagP'].shift(1+i)
        i += 1
        
    x.insert(loc=0, column='const',value=1)
    x.set_index(ind, inplace = True)
    
    # Prepping for OLS - break off as separate function
    lags = h + p -1
    y = y[lags:]
    x = x[lags:]
#    n = y.shape[0]
    
    #OlS - break off as separate function
    xT = x.T
    inv = pd.DataFrame(np.linalg.pinv(xT@x))
    xTy = xT@(np.asarray(y))
    betas = np.dot(inv, xTy)
    pred = x@betas
    ind2 = pred.index
    pred.rename(columns = {0:'Y'}, inplace=True)
    y.set_index(ind2, inplace = True)
    cycle = y.subtract(pred)
    cycle.rename(columns = {'Y': nm}, inplace=True)
    
    return pred, cycle

##############################################################################
#Boosted_HP_filter

def BoostedHP(x, lam = 1600, iter = True, stopping = "BIC", \
              sig_p = 0.050, Max_Iter = 100):
    
    
    x = np.array(x)
    
    ## generating trend operator matrix "S：        
    raw_x = x # save the raw data before HP
    n = len(x) # data size
    
    I_n = np.eye(n)
    D_temp = np.vstack((np.zeros([1,n]),np.eye(n-1,n)))
    D_temp= np.dot((I_n-D_temp),(I_n-D_temp))
    D = D_temp[2:n].T
    S = np.linalg.inv(I_n+lam*np.dot(D,D.T)) # Equation 4 in PJ
    mS = I_n - S
    
    ##########################################################################  
    
    ## the simple HP-filter
    if not iter:
        
        print("Original HP filter.")
        x_f = np.dot(S,x)
        x_c = x - x_f
        result = {"cycle": x_c, "trend_hist" : x_f, \
                   "stopping" : "nonstop", "trend" : x - x_c, "raw_data" : raw_x}
            
    ##########################################################################
            
    ## The Boosted HP-filter 
    if iter:
        ### ADF test as the stopping criterion
        if stopping == "adf":
            
            print("Boosted HP-ADF.")
            
            r = 1
            stationary = False
            x_c = x
            
            x_f = np.zeros([n,Max_Iter])
            adf_p = np.zeros([Max_Iter,1])
            
            while (r <= Max_Iter) and (not stationary):
                
                x_c = np.dot(mS,x_c)
                x_f[:,r-1] = x-x_c
                adf_p_r = ts.adfuller(x_c, maxlag = math.floor(pow(n-1,1/3)), autolag=None, \
                                      regression = "ct")[1]

                # x_c is the residual after the mean and linear trend being removed by HP filter
                # we use the critical value for the ADF distribution with
                    # the intercept and linear trend specification
                    
                adf_p[[r-1]] = adf_p_r
                stationary = adf_p_r <= sig_p
                
                # Truncate the storage matrix and vectors
                if stationary:
                    R = r
                    x_f = x_f[:,0:R]
                    adf_p = adf_p[0:R]
                    break
                
                r += 1
            
            if r > Max_Iter:
                R = Max_Iter
                logging.warning("The number of iterations exceeds Max_Iter. \
                The residual cycle remains non-stationary.")
                
            result = {"cycle" : x_c, "trend_hist" : x_f,  "stopping" : stopping,
                     "signif_p" : sig_p, "adf_p_hist" : adf_p, "iter_num" : R,
                    "trend" : x - x_c, "raw_data" : raw_x}
        
        
        else: # either BIC or nonstopping
            
            # assignment 
            r = 0
            x_c_r = x
            x_f = np.zeros([n,Max_Iter])
            IC = np.zeros([Max_Iter,1])
            # IC_decrease = True
            
            I_S_0 = I_n - S
            c_HP = np.dot(I_S_0, x)
            I_S_r = I_S_0
            
            while r < Max_Iter:
                
                r += 1
                
                x_c_r = np.dot(I_S_r, x)
                x_f[:,r-1] = x - x_c_r
                B_r = I_n - I_S_r 
                IC[[r-1]] =  np.var(x_c_r)/np.var(c_HP) + \
                    np.log(n)/(n-np.sum(np.diag(S))) * np.sum(np.diag(B_r))
                
                I_S_r = np.dot(I_S_0, I_S_r) # update for the next round
                
                if r >= 2 and stopping == "BIC":
                    if IC[[r-2]] < IC[[r-1]]:
                        break
            
            # final assignment
            R = r-1
            x_f = x_f[:, list(range(0,R))]
            x_c = x - x_f[:, R-1]
            
            if stopping == "BIC":
                
                print("Boosted HP-BIC.")
                # save the path of BIC till iter+1 times to keep the "turning point" of BIC history.
                result = {"cycle" : x_c, "trend_hist" : x_f,  "stopping" : stopping, 
                       "BIC_hist" : IC[0:(R+1)], "iter_num" : R, "trend" : x- x_c, "raw_data" : raw_x}
            
            if stopping == "nonstop":
                
                print('Boosted HP-BIC with stopping = "nonstop".')
                result = {"cycle" : x_c, "trend_hist" : x_f,  "stopping" : stopping, 
                       "BIC_hist" : IC, "iter_num" : Max_Iter - 1, "trend" : x- x_c, "raw_data" : raw_x}
            
    return result 

### function ends 
##############################################################################

# Main Streamlit app
def main():

    title_alignment = """ <style>
    .centered-title {
    text-align: center;}
    </style>
    <h1 class="centered-title">Time Series Filtering Methods</h1>
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

                
        st.write("### Preview of uploaded data:", df.head())

        # Select the dependent variable
        column  = st.selectbox("Select Column to filtering:", df.columns)

        st.title("Treating The Data")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Raw the Data", "Hodrick-Prescott Filter", "Christiano-Fitzgerald Filter", "Baxter-King Filter", 
                                                "Hamilton Filter", 'Boosted_HP_filter', "Plotting the filters"])


        # Apply filters:
        with tab1:
            st.header("This is Raw of uploading file")

            # Display the raw data
            st.write("Display the first few rows of the Data:", df.head())
            st.write("Display the last few rows of the Data: ",df.tail())

            st.write("Generate summary statistics: ",df.describe().T )
            
            
            data= df[column].dropna()
            
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.legend()
            plt.title(f'Plotting the Original Data : {column}' ,
                      color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 12)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)



        with tab2:
            # Hodrick-Prescott Filter
            st.subheader("Hodrick-Prescott Filter")
            st.markdown("""
                        **Quarterly Data:**
                        **(lambda = 1600)**: This is the most commonly used value for quarterly data and is widely accepted in macroeconomic applications.
                        
                        **Monthly Data:**
                        **(lambda = 14400)**: This value is often used for monthly data to account for the higher frequency of observations.
                        
                        **Annual Data:**
                        -   **(lambda = 100)**: According Hodrick & Prescott.   
                        -   **(lambda = 6.25)**: According Ravn & Uhlig, 2002.)
                        -   **(lambda = 10)**: According Baxter & King, 1995.)
                        """)
            
            
            values = [6.25, 10, 100, 1600]
            lamb = st.selectbox("Select a value for a smoothing parameter:", values)
            cycle_hp, trend_hp = sm.tsa.filters.hpfilter(data, lamb=lamb)
            results_hp = pd.DataFrame({'Cycle': cycle_hp, 'Trend': trend_hp, "Data": data})
            st.line_chart(results_hp)
            #########################################
            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(trend_hp, label='Trend', color='orange')
            ax.plot(cycle_hp, label='Cycle', color='green')

            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            plt.title(f'The filtered Data of {column} using Hodrick-Prescott Filter taking the value Lambda ={lamb}' ,
                      color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 12)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            # Save results to CSV
            csv = results_hp.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "hp_filter_results.csv", "text/csv")

            


        with tab3:


            # Christiano-Fitzgerald Filter
            st.subheader("Christiano-Fitzgerald Filter")
            st.markdown("""
                        Annual Data:
                        .   Low Frequency: 2
                        .   High Frequency: 8
                        
                        Semi-Annual Data:
                        .   Low Frequency: 3
                        .   High Frequency: 16
                        
                        Quarterly Data:
                        .   Low Frequency: 6
                        .   High Frequency: 32
                        
                        Monthly Data:
                        .   Low Frequency: 8
                        .   High Frequency: 96
                        
                        Weekly Data:
                        .   Low Frequency: 78
                        .   High Frequency: 416
                        
                        5 Days Data:
                        .   Low Frequency: 391.5
                        .   High Frequency: 2088
                        
                        7 Days Data:
                        .   Low Frequency: 547.5
                        .   High Frequency: 2920
                        """)
            low_values =[2, 3, 6, 8, 78, 391.5, 547.5 ]
            high_values =[8, 16, 32, 96, 416, 2088, 2920 ]
            low = st.selectbox("Select a Low Frequency:", low_values, key="low_cutoff") 
            high = st.selectbox("Select a High Frequency:", high_values, key="high_cutoff")
            cycle_cf, trend_cf = sm.tsa.filters.cffilter(data, low, high)
            results_cf = pd.DataFrame({'Cycle': cycle_cf, 'Trend': trend_cf, "Data": data})
            st.line_chart(results_cf)
            #########################################
            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(trend_cf, label='Trend', color='orange')
            ax.plot(cycle_cf, label='Cycle', color='green')
            plt.title('The filtered Data of {column} using Christiano-Fitzgerald Filter', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)

            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Save results to CSV
            csv = results_cf.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "CF_filter_results.csv", "text/csv")

        with tab4:   


            # Baxter-King Filter
            st.subheader("Baxter-King Filter")
            st.markdown("""
                        Annual Data:
                        .   Low Frequency: 2
                        .   High Frequency: 8
                        
                        Semi-Annual Data:
                        .   Low Frequency: 3
                        .   High Frequency: 16
                        
                        Quarterly Data:
                        .   Low Frequency: 6
                        .   High Frequency: 32
                        
                        Monthly Data:
                        .   Low Frequency: 8
                        .   High Frequency: 96
                        
                        Weekly Data:
                        .   Low Frequency: 78
                        .   High Frequency: 416
                        
                        5 Days Data:
                        .   Low Frequency: 391.5
                        .   High Frequency: 2088
                        
                        7 Days Data:
                        .   Low Frequency: 547.5
                        .   High Frequency: 2920

                        k: Lead-lag length of the filter. Baxter and King propose a truncation length of 12 for quarterly data and 3 for annual data.
                        """)
            low_values =[2, 3, 6, 8, 78 , 391.5, 547.5]
            high_values =[8, 16, 32, 96, 416 , 2088, 2920]
            k_value= [3, 12]
            low = st.selectbox("Select a Low Frequency:", low_values ) 
            high = st.selectbox("Select a High Frequency:", high_values)
            k= st.selectbox('Lead-lag length of the filter', k_value)

            cycle_bk  = sm.tsa.filters.bkfilter(data, low, high, K=k)
            results_bk = pd.DataFrame({'Cycle': cycle_bk, "Data": data})
            st.line_chart(results_bk)
            #########################################
            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(cycle_bk, label='Cycle', color='green')
            plt.title('The filtered Data of {column} using Baxter-King Filter', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)


            # Save results to CSV
            csv = results_bk.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "BK_filter_results.csv", "text/csv")

        with tab5:
            # Hamilton Filter
            st.subheader("Hamilton Filter")


            pred_ham, cycle_ham  = hamiltonFilter(data)

            pred_ham
            cycle_ham
            st.subheader("Trend Component")
            st.line_chart(pred_ham)

            st.subheader("Cycle Component")
            st.line_chart(cycle_ham)

            ##############################################
            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(pred_ham, label='Trend Component', color='green')
            ax.plot(cycle_ham, label='Cycle Component', color='orange')
            plt.title(f'The filtered Data of {column} using Hamilton Filter', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)




            # Ensure cycle and trend are Series
            results_ham = pd.DataFrame({'Trend': pred_ham['Y'], 'Cycle': cycle_ham[column], "Data": data })

            st.write("### The filtered data using Hamilton Filter:", results_ham.head())

            # Save results to CSV
            csv = results_ham.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "Hamilton_filter_results.csv", "text/csv")
            
        with tab6:

            st.subheader('Boosted Hodrick-Prescott Filter (bHP)')

            st.markdown("""
                        **Quarterly Data:**
                        **(lambda = 1600)**: This is the most commonly used value for quarterly data and is widely accepted in macroeconomic applications.
                        
                        **Monthly Data:**
                        **(lambda = 14400)**: This value is often used for monthly data to account for the higher frequency of observations.
                        
                        **Annual Data:**
                        -   **(lambda = 100)**: According Hodrick & Prescott.   
                        -   **(lambda = 6.25)**: According Ravn & Uhlig, 2002.)
                        -   **(lambda = 10)**: According Baxter & King, 1995.)
                        """)
            
            
            val = [6.25, 10, 100, 1600]
            lam = st.selectbox("Select a value for a smoothing parameter:", val, key='value')

            #' # raw HP filter

            bx_HP = BoostedHP(data, lam = lam, iter = False)
            bx_HP_cycle = bx_HP["cycle"].flatten()  # The cyclical component 
            bx_HP_trend = bx_HP["trend"].flatten()  # The trend component 

            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(bx_HP_trend, label='Trend Component', color='green')
            ax.plot(bx_HP_cycle, label='Cycle Component', color='orange')
            plt.title(f'The filtered Data of {column} using bHP Filter-raw', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Ensure cycle and trend are Series
            results_bx_HP = pd.DataFrame({'Trend': bx_HP_trend, 'Cycle': bx_HP_cycle})

            results_bx_HP.head()

            
            # Save results to CSV
            csv = results_bx_HP.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "bx_HP_filter_results.csv", "text/csv")

            
            
            #'################################################################################
            #' # by BIC
            
            bx_BIC = BoostedHP(data, lam = lam, iter = True, stopping = "BIC")
            bx_BIC_cycle = bx_BIC["cycle"].flatten()  # The cyclical component 
            bx_BIC_trend = bx_BIC["trend"].flatten()  # The trend component 

            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(bx_BIC_trend, label='Trend Component', color='green')
            ax.plot(bx_BIC_cycle, label='Cycle Component', color='orange')
            plt.title(f'The filtered Data of {column} using bHP Filter-BIC', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Ensure cycle and trend are Series
            

            results_bx_BIC = pd.DataFrame({'Trend': bx_BIC_trend, 'Cycle': bx_BIC_cycle})

            results_bx_BIC.head()

            
            # Save results to CSV
            csv = results_bx_BIC.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "bx_BIC_filter_results.csv", "text/csv")

#'################################################################################
            #' # by ADF
            
            bx_ADF = BoostedHP(data, lam = lam, iter = True, stopping = "adf")
            bx_ADF_cycle = bx_ADF["cycle"].flatten()  # The cyclical component 
            bx_ADF_trend = bx_ADF["trend"].flatten()  # The trend component 

            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(bx_ADF_trend, label='Trend Component', color='green')
            ax.plot(bx_ADF_cycle, label='Cycle Component', color='orange')
            plt.title(f'The filtered Data of {column} using bHP Filter-ADF', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Ensure cycle and trend are Series
            
            results_bx_ADF = pd.DataFrame({'Trend': bx_ADF_trend, 'Cycle': bx_ADF_cycle })

            
            # Save results to CSV
            csv = results_bx_ADF.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "bx_ADF_filter_results.csv", "text/csv")

            #'################################################################################

            #' # If stopping = "nonstop",
            bx_nonstop = BoostedHP(data, lam = lam, iter = True, stopping = "nonstop")
            bx_nonstop_cycle = bx_nonstop["cycle"].flatten()  # The cyclical component 
            bx_nonstop_trend = bx_nonstop["trend"].flatten()  # The trend component 

            plt.style.use('Solarize_Light2')
            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(data, label=f'The selected variables is : {column}')
            ax.plot(bx_nonstop_trend, label='Trend Component', color='green')
            ax.plot(bx_nonstop_cycle, label='Cycle Component', color='orange')
            plt.title(f'The filtered Data of {column} using bHP Filter-nonstop', color='darkslateblue',
                     font='georgia', 
                     fontweight= 'bold',
                     fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Ensure cycle and trend are Series
            results_bx_nonstop = pd.DataFrame({'Trend': bx_nonstop_trend, 'Cycle': bx_nonstop_cycle })

            
            # Save results to CSV
            csv = results_bx_nonstop.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV", csv, "bx_nonstop_filter_results.csv", "text/csv")

            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(bx_HP_trend, label='Trend Raw', color='green')
            ax.plot(bx_BIC_trend, label='Trend BIC', color='red')
            ax.plot(bx_ADF_trend, label='Trend ADF', color='purple')
            ax.plot(bx_nonstop_trend, label='Trend nonstop', color='blue')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'The filtered Trends of {column} using bHP Filter with Lamda : {lam}', color='darkslateblue',
                                font='georgia', 
                                fontweight= 'bold',
                                fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)

            st.pyplot(fig)

            fig, ax= plt.subplots( figsize=(10,8) , facecolor='lightblue')
            ax.plot(bx_HP_cycle, label='Cycle Raw', color='green')
            ax.plot(bx_BIC_cycle, label='Cycle BIC', color='red')
            ax.plot(bx_ADF_cycle, label='Cycle ADF', color='purple')
            ax.plot(bx_nonstop_cycle, label='Cycle nonstop', color='blue')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'The filtered Cycles of {column} using bHP Filter with Lamda : {lam}', color='darkslateblue',
                                font='georgia', 
                                fontweight= 'bold',
                                fontsize= 14)
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      facecolor='white', edgecolor='gray', fontsize=10)

            st.pyplot(fig)

        with tab7:

            st.markdown(" <h3 style='text-align: center; color: blue; font-size:18;'> Plotting the  filtered Data </h3>", unsafe_allow_html=True)


            # Create a 2x2 grid of subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 9), facecolor='lightblue' )

            # We can access each subplot by its index

            axs[0].plot(trend_hp, label='Trend using HP', color='orange')
            axs[0].plot(trend_cf, label='Trend using CF ', color='red')
            axs[0].plot(pred_ham, label='Trend using Hamilton', color='blue')
            axs[0].plot(bx_HP_trend, label='Trend using HP Raw', color='gray')
            axs[0].plot(bx_BIC_trend, label='Trend using bHP-BIC', color='purple')
            axs[0].plot(bx_ADF_trend, label='Trend using bHP-ADF', color='yellow')
            axs[0].plot(bx_nonstop_trend, label='Trend using bHP-nonstop', color='green')
            axs[0].set_title('Plotting the filtered Trends using difference methods ', fontsize=13, color='darkslateblue',
                             font='georgia', 
                             fontweight= 'bold')
            axs[0].tick_params( rotation=45)
            axs[0].legend(loc='upper right', frameon=True, framealpha=0.9, 
                         facecolor='white', edgecolor='gray', fontsize=8)
     


            axs[1].plot(cycle_hp, label='Cycle using HP', color='orange')
            axs[1].plot(cycle_cf, label='Cycle using CF', color='red')
            axs[1].plot(cycle_bk, label='Cycle using BK', color='black')
            axs[1].plot(cycle_ham, label='Cycle using Hamilton', color='blue')
            axs[1].plot(bx_HP_cycle, label='Cycle using HP Raw', color='gray')
            axs[1].plot(bx_BIC_cycle, label='Cycle using bHP-BIC', color='purple')
            axs[1].plot(bx_ADF_cycle, label='Cycle using bHP-ADF', color='yellow')
            axs[1].plot(bx_nonstop_cycle, label='Cycle using bHP-nonstop', color='green')
            axs[1].set_title('Plotting the filtered Cycle using difference methods ', fontsize=13, color='darkslateblue',
                             font='georgia', 
                             fontweight= 'bold')
            axs[1].tick_params( rotation=45)
            axs[1].legend(loc='upper right', frameon=True, framealpha=0.9, 
                         facecolor='white', edgecolor='gray', fontsize=8)

            
            # Add a shared title
            fig.suptitle(f'Filtering The Variable {column}', fontsize=16, color='darkblue',
                         font='georgia', 
                         fontweight= 'bold')
            st.pyplot(fig)


            st.markdown(" <h3 style='text-align: right; color: green; font-size:18;'> Download The filtered Data </h3>", unsafe_allow_html=True)

            results = pd.DataFrame({ 'Cycle hp': cycle_hp, 'Trend hp': trend_hp,
                        'Cycle Christiano-Fitzgerald': cycle_cf, 'Trend Christiano-Fitzgerald': trend_cf,
                        'Cycle Baxter & King': cycle_bk })
            
            csv = results.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV using HP BK CF", csv, "filter_results.csv", "text/csv")

            results_Bhp = pd.DataFrame({
                        'Cycle bx_HP': bx_HP_cycle, 'Trend bx_HP': bx_HP_trend,
                        'Cycle bx_BIC': bx_BIC_cycle, 'Trend bx_BIC': bx_BIC_trend,
                        'Cycle bx_ADF': bx_ADF_cycle, 'Trend bx_ADF': bx_ADF_trend,
                        'Cycle bx_nonstop': bx_nonstop_cycle, 'Trend bx_nonstop': bx_nonstop_trend
                        })
            csv = results_Bhp.to_csv(index=True)
            st.download_button("Download Cycle and Trend as CSV using Bhp", csv, "filter_results_Bhp.csv", "text/csv")





            

            

if __name__ == "__main__":
    main()