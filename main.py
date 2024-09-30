import os
import time
from datetime import datetime
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_test(file_path):
    """
    Load the multiple choice test questions. See the example file -- CDRE.py -- for the exam format.
    """
    return pd.read_csv(file_path)

def exam(system, user, model, max_retries=5, retry_delay=0.5):
    """
    Run the exam on the test. If an error occurs attempt completion again after {retry_delay} seconds
    to a maximum of {max_retries} times.
    """ 
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            end_time = time.time()
            latency = end_time - start_time
            answer = completion.choices[0].message.content
            return answer, latency
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1}/{max_retries} after a delay of {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached, moving to the next question.")
                return None, None

def run_test(test_data, num_runs, model):
    """Orchestration of the test and {num_runs}"""

    #TODO
    # Make this system prompt an editable argument so that 
    # different exam questions can be benchmarked.  
    system = """
    You are taking the Canadian Dietetic Registration Exam.
    A question will be given to you along with answer options.
    The options are numbered.
    Your response will only contain the number associated with the answer.
    """
    results = []
    for run in tqdm(range(num_runs), desc="Runs"):
        for _, question in tqdm(test_data.iterrows(), desc="Questions", leave=False, total=len(test_data)):
            user = f"{question['Context']}\n{question['Question']}\n{question['Options']}"
            answer, latency = exam(system, user, model) #returns answer string and latency
            #check if the returned answer is in the Answer column of the question set
            correct = str(question['Answer']) in answer if answer else False 
            results.append({
                'Run': run + 1,
                'Question': question['ID'],
                'Correct': int(correct), #convert bool to int to average easily
                'Latency': latency
            })
    return pd.DataFrame(results)

def save_results(results, output_file):
    """Write the test results to a .csv"""
    results.to_csv(output_file, index=False)

def main(test_file, num_runs, model):
    os.makedirs('results', exist_ok=True)
    test_data = load_test(os.path.join('tests', test_file)) #read questions to a dataframe
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_test(test_data, num_runs, model) #run test
    output_file = os.path.join('results', f"{model}_{os.path.splitext(test_file)[0]}_{now}_results.csv") #create file name
    save_results(results, output_file) #save the results
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example command: python3 main.py CDRE.csv --num_runs 10 --model gpt-4o
    # This runs the CDRE.csv exam questions through GPT-4o 10 times

    parser = argparse.ArgumentParser(description="Run exam benchmark tests")
    parser.add_argument("test_file", help="Name of the test file in the tests folder")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the test (default: 10)")
    parser.add_argument("--model", type=str, default='gpt-4o', help="The OpenAI model to use for this test.")
    args = parser.parse_args()
    main(args.test_file, args.num_runs, args.model)