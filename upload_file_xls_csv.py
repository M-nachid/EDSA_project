import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title("Upload CSV or Excel File and Plot Selected Columns"
"# Edited By Boussiala Mohamed Nachid"
         "boussiala.nachid@univ-alger3.dz")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Determine the file type and read the file accordingly
    if uploaded_file.name.endswith('.csv'):
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully!")
    elif uploaded_file.name.endswith('.xlsx') :
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        st.write("Excel file uploaded successfully!")


   
    # Display the DataFrame
    st.dataframe(df)

    # Get the list of columns
    columns = df.columns.tolist()

    # Multiselect widget for column selection
    selected_columns = st.multiselect("Select columns to plot:", columns)

    # Plotting the selected columns
    if selected_columns:
        st.write("You selected:", selected_columns)

        # Create a plot
        plt.figure(figsize=(10, 5))
        for column in selected_columns:
            plt.plot(df[column], label=column)

        plt.title("Selected Columns Plot")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        st.pyplot(plt)  # Display the plot in Streamlit
    else:
        st.write("Please select at least one column to plot.")
else:
    st.write("Please upload a file.")
