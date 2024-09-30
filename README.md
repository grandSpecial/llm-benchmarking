# Project: Exam Benchmark Testing with OpenAI

## Overview

This project is designed to benchmark the performance of OpenAI's language models by running them through multiple-choice test questions. It takes an exam file as input, runs the questions through an OpenAI model, evaluates the answer and saves the latency.

## Features

- Loads multiple-choice test questions from a CSV file.
- Sends the questions to an OpenAI model to generate responses.
- Handles retries for potential API errors.
- Records response times and correctness.
- Saves results in a CSV format for analysis.

## Installation

1. Clone this repository.
2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage

To run the benchmarking tool, use the following command:

```bash
python3 main.py <test_file> --num_runs <number_of_runs> --model <model_name>
```

### Arguments:

- `<test_file>`: The name of the CSV file containing the exam questions (must be in the `tests` folder).
- `--num_runs`: The number of times to run the test (default: 10).
- `--model`: The OpenAI model to use for this test (default: `gpt-4o`).

### Example:

```bash
python main.py CDRE.csv --num_runs 10 --model gpt-4o
```

This command runs the sample exam questions in `CDRE.csv` through the GPT-4o model 10 times.

## CSV File Format

The test file should be in CSV format with the following columns:

- `ID`: A unique identifier for each question.
- `Context`: Any context needed for the question.
- `Question`: The question text.
- `Options`: The answer options, typically numbered.
- `Answer`: The correct answer number.

## Project Structure

- `main.py`: The main script to run the benchmarking tests.
- `tests/`: Folder containing the multiple-choice question CSV files.
- `results/`: Folder where the test results CSV files are saved.
- `.env`: File to store your OpenAI API key. (copy .env.example to .env and add a key)

## Functionality

- `load_test(file_path)`: Loads test questions from a CSV file.
- `exam(system, user, model, max_retries, retry_delay)`: Runs a test question through the model and retries on error.
- `run_test(test_data, num_runs, model)`: Orchestrates running all test questions for the specified number of runs.
- `save_results(results, output_file)`: Saves the results to a CSV file.
- `main(test_file, num_runs, model)`: Main entry point for running the benchmark tests.

## Requirements

- Python 3.7+
- OpenAI Python SDK
- Pandas
- tqdm
- python-dotenv

## License

This project is licensed under the MIT License.