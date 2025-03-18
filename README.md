# Project: Exam Benchmark Testing with LLM Providers

## Overview

This project benchmarks the performance of various language model providers' flagship models, such as OpenAI, Anthropic, and Google, by running them through multiple-choice test questions. It evaluates response accuracy and latency, providing comparative insights into each provider’s model performance.

## Features

- Loads multiple-choice test questions from a CSV file.
- Sends questions to the selected LLM provider’s flagship model.
- Manages retries for potential API errors.
- Records response times and correctness.
- Saves results in a CSV format for analysis.

## Installation

1. Clone this repository.
2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add API keys for each LLM provider:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    GROK_API_KEY=xxx
    GROK_API_KEY=xxx #llama's provider 
    DEEPSEEK_API_KEY=xxx
    ```

## Usage

To run the benchmarking tool, use the following command:

```bash
python3 main.py <test_file> --num_runs <number_of_runs> --provider <provider_name> --temperature <int:temperature>
```

### Arguments:

- `<test_file>`: The name of the CSV file containing the exam questions (must be in the `tests` folder).
- `--num_runs`: The number of times to run the test (default: 10).
- `--provider`: The LLM provider to use for this test (options: `openai`, `anthropic`, `google`, `etc.`).
- `--temperature`: The temperature for the LLM (less (0) or more (1) random)

### Example:

```bash
python3 main.py CDRE.csv --num_runs 10 --provider openai --temperature 0
```

This command runs the sample exam questions in `CDRE.csv` through OpenAI’s flagship model 10 times.

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
- `.env`: File to store API keys for each LLM provider.

## Functionality

- `load_test(file_path)`: Loads test questions from a CSV file.
- `exam(system, user, provider, max_retries, retry_delay)`: Runs a test question through the selected LLM provider's model and retries on error.
- `run_test(test_data, num_runs, provider)`: Orchestrates running all test questions for the specified number of runs.
- `save_results(results, output_file)`: Saves the results to a CSV file.
- `main(test_file, num_runs, provider)`: Main entry point for running the benchmark tests.

## Requirements

- Python 3.7+
- OpenAI, Anthropic, and Google SDKs
- Pandas
- tqdm
- python-dotenv
- & others. See requirements.txt 

`$ pip install -r requirements.txt`

## License

This project is licensed under the MIT License.

## Disclosure
The clean parts of this file were generated using GPT-4o 