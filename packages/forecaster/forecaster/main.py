"""Main entry point for the forecasting platform."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from forecaster.interface import ForecastingConversation

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Try default location


def main():
    """Simple CLI interface."""
    print("Forecaster - AI-Powered Time Series Forecasting")
    print("=" * 50)

    conversation = ForecastingConversation()

    # Example usage
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"\nLoading data from: {data_path}")

        load_result = conversation.load_data(data_path)
        if not load_result["success"]:
            print(f"Error: {load_result['message']}")
            sys.exit(1)

        print(f"Loaded {load_result['n_points']} data points")
        print(f"Validation: {json.dumps(load_result['validation'], indent=2)}")

        # Example forecast request
        if len(sys.argv) > 2:
            user_request = " ".join(sys.argv[2:])
        else:
            user_request = "Forecast the next 7 days"

        print(f"\nProcessing request: {user_request}")
        forecast_result = conversation.request_forecast(user_request)

        if forecast_result["success"]:
            print("\nForecast Results:")
            print(json.dumps(forecast_result["forecast"], indent=2))

            if forecast_result.get("warnings"):
                print("\nWarnings:")
                for warning in forecast_result["warnings"]:
                    print(f"  - {warning}")
        else:
            print(f"Error: {forecast_result['message']}")
            sys.exit(1)
    else:
        print("\nUsage: python -m forecaster.main <data_path> [forecast_request]")
        print("\nExample:")
        print("  python -m forecaster.main data.csv 'Forecast next 30 days'")
        sys.exit(0)


if __name__ == "__main__":
    main()
