"""Demographics service for zipcode-based demographic data lookup."""

import pandas as pd
from typing import Dict, List


class ZipcodeNotFoundError(Exception):
    """Raised when a zipcode is not found in demographics data."""
    pass


class DemographicsService:
    """In-memory cache for zipcode demographics (70 rows = ~28KB)."""

    def __init__(self, csv_path: str):
        """Load demographics CSV into memory for O(1) lookup.

        Args:
            csv_path: Path to zipcode_demographics.csv file
        """
        # Load CSV into pandas DataFrame
        self._df = pd.read_csv(csv_path, dtype={'zipcode': str})

        # Convert to dict for O(1) lookup: {zipcode: {column: value}}
        self._demographics_dict = {}
        for _, row in self._df.iterrows():
            zipcode = row['zipcode']
            # Convert row to dict, excluding the zipcode column itself
            demographics = row.drop('zipcode').to_dict()
            self._demographics_dict[zipcode] = demographics

        # Validate data loaded
        if len(self._demographics_dict) == 0:
            raise ValueError(f"No demographics loaded from {csv_path}")

    def get_demographics(self, zipcode: str) -> Dict[str, float]:
        """Get demographics for a given zipcode.

        Args:
            zipcode: 5-digit zipcode string

        Returns:
            Dictionary of demographic features for the zipcode

        Raises:
            ZipcodeNotFoundError: If zipcode not found in data
        """
        if zipcode not in self._demographics_dict:
            raise ZipcodeNotFoundError(
                f"Zipcode {zipcode} not found in demographics data. "
                f"Available zipcodes: {len(self._demographics_dict)}"
            )

        return self._demographics_dict[zipcode]

    def get_available_zipcodes(self) -> List[str]:
        """Return list of all available zipcodes.

        Returns:
            List of zipcode strings
        """
        return list(self._demographics_dict.keys())
