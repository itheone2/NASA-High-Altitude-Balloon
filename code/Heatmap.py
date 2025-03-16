#Heatmap 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns.tolist())
    print("\nDataset shape:", df.shape)
    return df

def preprocess_data(df):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time_actual'])

    # Convert to Mountain Time
    mountain = pytz.timezone('US/Mountain')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(mountain)

    # Drop non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    return numeric_df

def create_heatmap(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(30, 25))

    # Create a heatmap using seaborn
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Customize the plot
    plt.title('Correlation Heatmap of All Numeric Variables', fontsize=20)
    plt.tight_layout()

    # Save the plot
    plt.savefig('comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as 'comprehensive_heatmap.png'")

def main():
    file_path = 'cleaned_gps_data.csv'
    df = load_data(file_path)
    numeric_df = preprocess_data(df)
    create_heatmap(numeric_df)

if __name__ == "__main__":
    main()