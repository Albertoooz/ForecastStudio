"""Data Transformer Agent — manipulates and transforms data (Polars)."""

from pathlib import Path
from typing import Any

import polars as pl


class DataTransformer:
    """
    Data Transformer — performs data manipulations.

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
            df = self._load_file(filepath)

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

            combined = (
                pl.col(date_column).cast(pl.Utf8) + pl.lit(" ") + pl.col(time_column).cast(pl.Utf8)
            ).str.to_datetime(strict=False)
            df = df.with_columns(combined.alias(output_column))

            new_filepath = filepath.parent / f"{filepath.stem}_transformed{filepath.suffix}"
            df.write_csv(new_filepath)

            from forecaster.data.analyzer import analyze_file

            data_info = analyze_file(new_filepath, new_filepath.name)
            data_info.datetime_column = output_column

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

    def _load_file(self, filepath: Path) -> pl.DataFrame:
        suffix = filepath.suffix.lower()
        if suffix == ".csv":
            return pl.read_csv(filepath)
        if suffix in (".xlsx", ".xls"):
            return pl.read_excel(filepath)
        return pl.read_csv(filepath)
