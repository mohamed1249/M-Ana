from pathlib import Path

import numpy as np
import pandas as pd

from MAna.analysis import quick_dashboard
from MAna.data import DataCleaner, read_data


ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = ROOT / "docs" / "notebooks" / "data" / "womens_clothing_reviews_sample.csv"


reviews = read_data(REVIEWS_PATH)
reviews = (
    DataCleaner(reviews, verbose=False)
    .remove_duplicates()
    .fix_missing_values(
        strategy={
            "Age": "median",
            "Title": "mode",
            "Review Text": "mode",
            "Division Name": "mode",
            "Department Name": "mode",
            "Class Name": "mode",
        }
    )
    .get_cleaned_data()
)

reviews["Review Length"] = reviews["Review Text"].fillna("").str.split().str.len()
reviews["Recommended Label"] = np.where(
    reviews["Recommended IND"].eq(1),
    "Recommended",
    "Not Recommended",
)

dashboard = quick_dashboard(
    reviews,
    title="Quick Clothing Reviews Dashboard",
    numerical_cols=["Age", "Rating", "Positive Feedback Count", "Review Length"],
    categorical_cols=["Department Name", "Recommended Label"],
    port=8050,
    debug=False,
    run=True,
)
