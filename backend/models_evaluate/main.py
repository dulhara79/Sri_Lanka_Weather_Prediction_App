from weatherModelEvaluator import WeatherModelEvaluator
# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = WeatherModelEvaluator()
    
    # Option 1: Run complete evaluation
    # print("Place your test data CSV file in the same directory and update the path below:")
    # test_data_path = 'your_test_data.csv'  # Update this path
    
    # try:
    #     results, predictions = evaluator.run_complete_evaluation(weather_test_data.csv)
    # except FileNotFoundError:
    #     print("Test data file not found. Please update the path in the script.")
    # except Exception as e:
    #     print(f"Error during evaluation: {e}")
    
    # Option 2: Step-by-step evaluation
    evaluator.load_models()
    X_test, y_test, df_test = evaluator.load_test_data('weather_test_data.csv')
    results, predictions = evaluator.evaluate_all_models(X_test, y_test.values)
    evaluator.print_detailed_results(results)
    evaluator.create_evaluation_report(results)
    evaluator.create_prediction_comparison(y_test.values, predictions)