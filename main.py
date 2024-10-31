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
import itertools
import json

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

def load_prompts(file_path):
    """Load system prompts from a JSON file"""
    with open(file_path, 'r') as f:
        prompts = json.load(f)
    return list(prompts.values())

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

def run_test(test_data, num_runs, provider, temperature, system_prompt):
    """Orchestration of the test and {num_runs}"""
    results = []
    for run in tqdm(range(num_runs), desc="Runs"):
        for _, question in tqdm(test_data.iterrows(), desc="Questions", leave=False, total=len(test_data)):
            user = f"{question['Context']}\n{question['Question']}\n{question['Options']}"
            answer, latency = exam(system_prompt, user, provider, temperature) #returns answer string and latency
            #check if the returned answer is in the Answer column of the question set
            correct = str(question['Answer']) in answer if answer else False 
            results.append({
                'Run': run + 1,
                'Question': question['ID'],
                'Provider': provider,
                'Temperature': temperature,
                'System Prompt': system_prompt,
                'Correct': int(correct), #convert bool to int to average easily
                'Latency': latency,
                'Generated_Response': answer[0] if answer else None
            })
    return pd.DataFrame(results)

def save_results(results, output_file):
    """Write the test results to a .csv"""
    results.to_csv(output_file, index=False)

def main(test_file, num_runs, providers, temperatures):
    os.makedirs('results', exist_ok=True)
    test_data = load_test(os.path.join('tests', test_file)) #read questions to a dataframe
    prompt_file = os.path.join('tests', 'system_prompts.json') # Load prompts from the tests folder
    system_prompts = load_prompts(prompt_file)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    # Iterate over all combinations of providers, temperatures, and system prompts
    for provider, temperature, system_prompt in itertools.product(providers, temperatures, system_prompts):
        print(f"Running test with provider: {provider}, temperature: {temperature}, system prompt: {system_prompt[:50]}...")
        results = run_test(test_data, num_runs, provider, temperature, system_prompt) #run test
        all_results.append(results)
    # Concatenate all results into a single dataframe
    final_results = pd.concat(all_results, ignore_index=True)
    output_file = os.path.join('results', f"combined_results_{now}.csv") #create file name
    save_results(final_results, output_file) #save the results
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example command: python3 main.py CDRE.csv --num_runs 10 --providers openai anthropic --temperatures 0 0.5
    # This runs the CDRE.csv exam questions through multiple configurations

    parser = argparse.ArgumentParser(description="Run exam benchmark tests")
    parser.add_argument("test_file", help="Name of the test file in the tests folder")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the test (default: 10)")
    parser.add_argument("--providers", nargs='+', help="List of LLM providers to use for this test.")
    parser.add_argument("--temperatures", nargs='+', type=float, help="List of temperatures to use for this test.")
    args = parser.parse_args()
    main(args.test_file, args.num_runs, args.providers, args.temperatures)
