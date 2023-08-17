import pandas as pd
import os
def analyze_status(file_path):
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Count the occurrences of each category in the 'status' column
        status_counts = df['status'].value_counts().to_dict()

        return status_counts
    except Exception as e:
        return str(e) 

def analyze_label(file_path):
    df = pd.read_csv(file_path)
    return df['label'].value_counts().to_dict()

def summarize_csv(file_path):
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Get column names
        columns = df.columns
        
        # Get number of data points (rows)
        num_data_points = len(df)
        
        # Check for missing data points and count them for each column
        missing_data_points = df.isnull().sum()
        
        # Filter out columns without missing values
        missing_data_points = missing_data_points[missing_data_points > 0]
        
        return {
            "Columns": columns,
            "Number of Data Points": num_data_points,
            "Missing Data Points": missing_data_points
        }
    except Exception as e:
        return str(e)

# Analyze files for each phase
phases = ["I", "II", "III"]
file_types = ["train", "valid", "test"]
# Print columns once
print("Columns: nctid, status, why_stop, label, phase, diseases, icdcodes, drugs, smiless, criteria\n") 

for phase in phases:
    print(f"\nSummary for Phase {phase}:")
    for file_type in file_types:
        file_name = f"phase_{phase}_{file_type}.csv"
        
        # Check if file exists
        if os.path.exists(file_name):
            print(f"\nSummary for {file_name}:")
            summary = summarize_csv(file_name)
            
            # Display Summary
            if "Columns" in summary:
                #print(f"Columns: {', '.join(summary['Columns'])}")
                print(f"Number of Data Points: {summary['Number of Data Points']}")
                if len(summary['Missing Data Points']) > 0:
                    for column, missing_count in summary['Missing Data Points'].items():
                        print(f"Missing Data Points in '{column}': {missing_count}")
                else:
                    print("No Missing Data Points!")
            else:
                print(f"Error: {summary}")
        else:
            print(f"{file_name} does not exist!")

        # Check if file exists
        if os.path.exists(file_name):
            print(f"\nStatus Summary for {file_name}:")
            status_counts = analyze_status(file_name)
            labels = analyze_label(file_name)

            # Display Status Summary
            if isinstance(status_counts, dict):
                for status, count in status_counts.items():
                    print(f"{status}: {count}")
                
                print("\nLabel Counts:")
                for label, count in labels.items():
                    print(f"{label}: {count}") 
            else:
                print(f"Error: {status_counts}")
        else:
            print(f"{file_name} does not exist!")

