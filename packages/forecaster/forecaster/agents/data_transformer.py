"""Data Transformer Agent - manipulates and transforms data."""

from pathlib import Path
from typing import Any

import pandas as pd


class DataTransformer:
    """
    Data Transformer - performs data manipulations.

    Responsibilities:
    - Combine columns (e.g., date + time)
    - Transform data types
    - Create derived columns
    - Clean data
    """

    def combine_datetime_columns(
        self,
        filepath: Path,
        date_column: str,
        time_column: str,
        output_column: str = "datetime",
    ) -> dict[str, Any]:
        """
        Combine date and time columns into a single datetime column.

        Args:
            filepath: Path to data file
            date_column: Name of date column (e.g., 'business_date')
            time_column: Name of time column (e.g., 'time_15min_slot')
            output_column: Name for the new combined column

        Returns:
            {
                "success": bool,
                "message": str,
                "new_filepath": Path (if success),
                "data_info": DataInfo (if success)
            }
        """
        try:
            # Load file
            df = self._load_file(filepath)

            # Validate columns exist
            if date_column not in df.columns:
                return {
                    "success": False,
                    "message": f"Kolumna '{date_column}' nie istnieje",
                    "new_filepath": None,
                    "data_info": None,
                }

            if time_column not in df.columns:
                return {
                    "success": False,
                    "message": f"Kolumna '{time_column}' nie istnieje",
                    "new_filepath": None,
                    "data_info": None,
                }

            # Combine columns - handle different date formats
            # Convert date column to string if needed
            date_str = df[date_column].astype(str)
            time_str = df[time_column].astype(str)

            # Combine: "2024-01-01" + " " + "22:30" = "2024-01-01 22:30"
            combined_str = date_str + " " + time_str

            # Parse to datetime
            df[output_column] = pd.to_datetime(
                combined_str,
                format="mixed",
                errors="coerce",
            )

            # Check if parsing was successful
            if df[output_column].isna().sum() > len(df) * 0.1:  # More than 10% failed
                # Try alternative format
                df[output_column] = pd.to_datetime(
                    combined_str,
                    infer_datetime_format=True,
                    errors="coerce",
                )

            # Drop original columns if requested (optional)
            # df = df.drop(columns=[date_column, time_column])

            # Save to new file
            new_filepath = filepath.parent / f"{filepath.stem}_transformed{filepath.suffix}"
            df.to_csv(new_filepath, index=False)

            # Re-analyze the transformed file
            from forecaster.data.analyzer import analyze_file

            data_info = analyze_file(new_filepath, new_filepath.name)
            data_info.datetime_column = output_column  # Set as datetime column

            return {
                "success": True,
                "message": f"Połączono kolumny '{date_column}' i '{time_column}' w '{output_column}'",
                "new_filepath": new_filepath,
                "data_info": data_info,
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Błąd podczas łączenia kolumn: {str(e)}",
                "new_filepath": None,
                "data_info": None,
            }

    def _load_file(self, filepath: Path) -> pd.DataFrame:
        """Load file into DataFrame."""
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(filepath)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(filepath)
        else:
            return pd.read_csv(filepath)
