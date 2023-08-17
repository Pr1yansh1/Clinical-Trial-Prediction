import pandas as pd
import os

def analyze_status(df):
    return df['status'].value_counts().to_dict()

def analyze_label(df):
    return df['label'].value_counts().to_dict()

def analyze_combined_phase(phase_files):
    try:
        # Combine CSV files
        combined_df = pd.concat([pd.read_csv(file) for file in phase_files])

        # Count the occurrences of each category in the 'status' column
        status_counts = analyze_status(combined_df)

        # Count the occurrences of each category in the 'label' column
        label_counts = analyze_label(combined_df)

        # Get number of data points (rows)
        num_data_points = len(combined_df)

        # Check for missing data points and count them for each column
        missing_data_points = combined_df.isnull().sum()

        # Filter out columns without missing values
        missing_data_points = missing_data_points[missing_data_points > 0]

        return {
            "Status Counts": status_counts,
            "Label Counts": label_counts,
            "Number of Data Points": num_data_points,
            "Missing Data Points": missing_data_points
        }
    except Exception as e:
        return str(e)

# Print columns once
print("Columns: nctid, status, why_stop, label, phase, diseases, icdcodes, drugs, smiless, criteria\n")

# Analyze files for each phase
phases = ["I", "II", "III"]
file_types = ["train", "valid", "test"]

for phase in phases:
    phase_files = [f"phase_{phase}_{file_type}.csv" for file_type in file_types]
    existing_phase_files = [file for file in phase_files if os.path.exists(file)]
    
    if not existing_phase_files:
        print(f"\nNo files found for Phase {phase}.")
        continue

    print(f"\nCombined Summary for Phase {phase}:")
    analysis = analyze_combined_phase(existing_phase_files)

    # Display Analysis
    if "Status Counts" in analysis:
        print(f"Number of Data Points: {analysis['Number of Data Points']}")

        print("\nStatus Counts:")
        for status, count in analysis["Status Counts"].items():
            print(f"{status}: {count}")

        print("\nLabel Counts:")
        for label, count in analysis["Label Counts"].items():
            print(f"{label}: {count}")

        if len(analysis['Missing Data Points']) > 0:
            print("\nMissing Data Points:")
            for column, missing_count in analysis['Missing Data Points'].items():
                print(f"'{column}': {missing_count}")
        else:
            print("No Missing Data Points!")
    else:
        print(f"Error: {analysis}")
