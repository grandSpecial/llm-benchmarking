import os
import time
from datetime import datetime
from openai import OpenAI
import anthropic
import google.generativeai as g_client
from google.generativeai.types import GenerationConfig
import pandas as pd
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

o_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
a_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
g_client.configure(api_key=os.getenv('GEMINI_API_KEY'))

def gpt(system,user,temperature):
  """Make request to OpenAI flagship"""
  start_time = time.time()
  completion = o_client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": system},
          {"role": "user", "content": user}
      ],
      max_tokens=2048,
      temperature=int(temperature),
  )
  end_time = time.time()
  latency = end_time - start_time
  answer = completion.choices[0].message.content
  return answer, latency

def claude(system,user,temperature):
  """Make request to Anthropic flagship"""
  start_time = time.time()
  message = a_client.messages.create(
      model="claude-3-5-sonnet-20241022",
      max_tokens=2048,
      temperature=int(temperature),
      system = system,
      messages=[
          {"role": "user", "content": user}
      ]
  )
  end_time = time.time()
  latency = end_time - start_time
  answer = message.content[0].text
  return answer, latency 

def gemini(system,user,temperature):
  """Make request to Google flagship"""
  start_time = time.time()
  model = g_client.GenerativeModel(
    "gemini-1.5-pro", 
    generation_config=GenerationConfig(temperature=int(temperature)),
    system_instruction=system
  )
  response = model.generate_content(user)
  end_time = time.time()
  latency = end_time - start_time
  return response.text, latency

def load_test(file_path):
    """
    Load the multiple choice test questions. See the example file -- CDRE.py -- for the exam format.
    """
    return pd.read_csv(file_path)

def exam(system, user, provider, temperature, max_retries=5, retry_delay=0.5):
    """
    Run the exam on the test. If an error occurs attempt completion again after {retry_delay} seconds
    to a maximum of {max_retries} times.
    """ 
    for attempt in range(max_retries):
        try:
            if provider == 'openai':
                response = gpt(system,user,temperature)
            elif provider == 'anthropic':
                response = claude(system,user,temperature)
            elif provider == 'google':
                response = gemini(system,user,temperature)
            else:
                return "Not a valid model. Please select one of openai, anthropic or google."
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1}/{max_retries} after a delay of {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached, moving to the next question.")
                return None, None

def run_test(test_data, num_runs, provider, temperature):
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
            answer, latency = exam(system, user, provider, temperature) #returns answer string and latency
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

def main(test_file, num_runs, provider, temperature):
    os.makedirs('results', exist_ok=True)
    test_data = load_test(os.path.join('tests', test_file)) #read questions to a dataframe
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_test(test_data, num_runs, provider, temperature) #run test
    output_file = os.path.join('results', f"{provider}_{os.path.splitext(test_file)[0]}_{now}_results.csv") #create file name
    save_results(results, output_file) #save the results
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example command: python3 main.py CDRE.csv --num_runs 10 --provider openai
    # This runs the CDRE.csv exam questions through openai's flagship LLM 10 times

    parser = argparse.ArgumentParser(description="Run exam benchmark tests")
    parser.add_argument("test_file", help="Name of the test file in the tests folder")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the test (default: 10)")
    parser.add_argument("--provider", type=str, default='openai', help="The LLM provider to use for this test.")
    parser.add_argument("--temperature", type=str, default=0, help="The termperature to use for this test.")
    args = parser.parse_args()
    main(args.test_file, args.num_runs, args.provider, args.temperature)