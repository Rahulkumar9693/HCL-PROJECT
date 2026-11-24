
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. FIND AND LOAD DATASET
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

csv_files = list(BASE_DIR.glob("*student_performance*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "CSV file not found! Put your student_performance CSV "
        "inside the same folder as this Python file."
    )

csv_path = csv_files[0]
df = pd.read_csv(csv_path)

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("File:", csv_path.name)
print("Rows and Columns:", df.shape)
print("Available columns:", list(df.columns))


# =====================================================
# 2. HELPER FUNCTION
# =====================================================

def find_col(keyword):
    matches = [
        col for col in df.columns
        if keyword.lower() in col.lower()
    ]

    if matches:
        return matches[0]

    return None


# =====================================================
# 3. IDENTIFY IMPORTANT COLUMNS
# =====================================================

id_col = find_col("student_id")
attendance_col = find_col("attendance")
study_col = find_col("study_hours")
grade_col = find_col("grade")

score_cols = [
    col for col in df.columns
    if ("score" in col.lower() or "marks" in col.lower())
    and col.lower() not in [
        "total_marks", "average_marks"
    ]
]

# Convert score columns to numeric
for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].clip(0, 100)

# Calculate total and average marks
if score_cols:
    df["Total_Marks"] = df[score_cols].sum(axis=1)
    df["Average_Marks"] = df[score_cols].mean(axis=1)
else:
    raise ValueError("No score or marks columns found.")

# Attendance cleaning
if attendance_col:
    df[attendance_col] = pd.to_numeric(
        df[attendance_col], errors="coerce"
    )
    df[attendance_col] = df[attendance_col].fillna(
        df[attendance_col].median()
    )

# Study hours cleaning
if study_col:
    df[study_col] = pd.to_numeric(
        df[study_col], errors="coerce"
    )


# =====================================================
# 4. CREATE GRADE COLUMN
# =====================================================

bins = [-1, 39.99, 59.99, 74.99, 89.99, 100]

labels = [
    "F (<40)",
    "D (40-59.99)",
    "C (60-74.99)",
    "B (75-89.99)",
    "A (90-100)"
]

df["Grade"] = pd.cut(
    df["Average_Marks"],
    bins=bins,
    labels=labels
)

# At-risk students
df["Academic_Status"] = np.where(
    df["Average_Marks"] < 40,
    "At-Risk",
    "Pass"
)


# =====================================================
# 5. PRINT PROJECT RESULTS
# =====================================================

print("\n========== STUDENT PERFORMANCE ANALYSIS ==========")

print("\n1. DATASET OVERVIEW")
print(df.head())

print("\n2. MISSING VALUES")
print(df.isnull().sum())

if id_col:
    print("\n3. DUPLICATE STUDENT IDs")
    print(df[id_col].duplicated().sum())

print("\n4. GRADE DISTRIBUTION")
print(df["Grade"].value_counts().sort_index())

print("\n5. TOP 10 STUDENTS")
if id_col:
    print(
        df.nlargest(10, "Average_Marks")[
            [id_col, "Total_Marks", "Average_Marks", "Grade"]
        ]
    )

print("\n6. AT-RISK STUDENTS")
print(
    df[df["Academic_Status"] == "At-Risk"][
        [id_col, "Average_Marks"]
        if id_col else ["Average_Marks"]
    ]
)

if study_col:
    print("\n7. STUDY HOURS VS AVERAGE MARKS")
    print(
        df[[study_col, "Average_Marks"]].corr().round(2)
    )

if attendance_col:
    print("\n8. ATTENDANCE VS AVERAGE MARKS")
    print(
        df[[attendance_col, "Average_Marks"]].corr().round(2)
    )

print("\n9. PYTHON STATISTICS")
python_col = find_col("python")

if python_col:
    values = df[python_col].dropna().to_numpy()

    print("Median:", np.median(values))
    print("Standard Deviation:", np.std(values, ddof=1))
    print("Quartiles:", np.percentile(values, [25, 50, 75]))


# =====================================================
# 6. SAVE CLEANED DATASET
# =====================================================

df.to_csv(
    BASE_DIR / "cleaned_student_performance.csv",
    index=False
)


