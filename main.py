import os
import time
import asyncio
import pandas as pd
from datetime import datetime
from openai import OpenAI
import anthropic
import google.generativeai as g_client
from google.generativeai.types import GenerationConfig
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
import aiofiles
import requests
from groq import Groq

# Load environment variables
load_dotenv()

o_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
a_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
g_client.configure(api_key=os.getenv('GEMINI_API_KEY'))
d_client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
gq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
grok_api_key = os.getenv("GROK_API_KEY")

async def gpt(system, user, temperature):
    """Async request to OpenAI."""
    start_time = time.time()
    try:
        completion = await asyncio.to_thread(
            o_client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=2048,
            temperature=float(temperature),
        )
        end_time = time.time()
        latency = end_time - start_time
        answer = completion.choices[0].message.content
        return answer, latency
    except Exception as e:
        print(f"Error in OpenAI request: {e}")
        return None, None

async def claude(system, user, temperature):
    """Async request to Anthropic."""
    start_time = time.time()
    try:
        message = await asyncio.to_thread(
            a_client.messages.create,
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            temperature=float(temperature),
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        end_time = time.time()
        latency = end_time - start_time
        answer = message.content[0].text
        return answer, latency
    except Exception as e:
        print(f"Error in Anthropic request: {e}")
        return None, None

async def gemini(system, user, temperature):
    """Async request to Google Gemini."""
    start_time = time.time()
    try:
        model = g_client.GenerativeModel(
            "gemini-1.5-pro", 
            generation_config=GenerationConfig(temperature=float(temperature)),
            system_instruction=system
        )
        response = await asyncio.to_thread(model.generate_content, user)
        end_time = time.time()
        latency = end_time - start_time
        return response.text, latency
    except Exception as e:
        print(f"Error in Google Gemini request: {e}")
        return None, None
    
async def grok(system, user, temperature):
    """Async request to Grok API."""
    start_time = time.time()
    try:
        headers = {
            "Authorization": f"Bearer {grok_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "model": "grok-2-1212",
            "temperature": float(temperature),
            "max_tokens": 2048
        }
        response = await asyncio.to_thread(
            requests.post,
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        end_time = time.time()
        latency = end_time - start_time
        answer = response.json()['choices'][0]['message']['content']
        return answer, latency
    except Exception as e:
        print(f"Error in Grok request: {e}")
        return None, None

async def llama(system, user, temperature):
    """Async request to Groq Llama 3."""
    start_time = time.time()
    try:
        completion = gq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user","content": user},
                {"role": "system", "content": system}
            ],
            temperature=float(temperature),
            max_completion_tokens=2048,
        )
        end_time = time.time()
        latency = end_time - start_time
        answer = completion.choices[0].message.content
        return answer, latency
    except Exception as e:
        print(f"Error in Groq Llama 3 request: {e}")
        return None, None

async def deepseek(system, user, temperature):
    """Async request to DeepSeek."""
    start_time = time.time()
    try:
        completion = await asyncio.to_thread(
            d_client.chat.completions.create,
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=2048,
            temperature=float(temperature),
        )
        end_time = time.time()
        latency = end_time - start_time
        answer = completion.choices[0].message.content
        return answer, latency
    except Exception as e:
        print(f"Error in DeepSeek request: {e}")
        return None, None

async def exam(system, user, provider, temperature, max_retries=5):
    """Run a single exam query with retries."""
    delay = 1  # Initial retry delay
    for attempt in range(max_retries):
        try:
            if provider == 'openai':
                return await gpt(system, user, temperature)
            elif provider == 'anthropic':
                return await claude(system, user, temperature)
            elif provider == 'google':
                return await gemini(system, user, temperature)
            elif provider == 'llama':
                return await llama(system, user, temperature)
            elif provider == 'deepseek':
                return await deepseek(system, user, temperature)
            elif provider == 'grok':
                return await grok(system, user, temperature)
            else:
                print(f"Invalid provider: {provider}")
                return None, None
        except Exception as e:
            print(f"Error in exam attempt {attempt + 1}: {e}")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
    print("Maximum retries reached.")
    return None, None

async def process_question(run, question, system, provider, temperature):
    """Process a single question."""
    user = f"{question['Context']}\n{question['Question']}\n{question['Options']}"
    answer, latency = await exam(system, user, provider, temperature)
    correct = str(question['Answer']) in answer if answer else False
    return {'Run': run + 1, 'Question': question['ID'], 'Correct': int(correct), 'Latency': latency}

async def run_test(test_data, num_runs, provider, temperature, checkpoint_file):
    """Run all tests sequentially and save progress iteratively."""
    system = """
    You are taking the Canadian Dietetic Registration Exam.
    A question will be given to you along with answer options.
    The options are numbered.
    Your response will only contain the number associated with the answer.
    """
    results = []
    if os.path.exists(checkpoint_file):
        results = pd.read_csv(checkpoint_file).to_dict('records')

    async with aiofiles.open(checkpoint_file, 'w') as f:
        await f.write("Run,Question,Correct,Latency\n")

    for run in tqdm(range(num_runs), desc="Runs"):
        run_results = []
        for _, question in tqdm(test_data.iterrows(), desc=f"Processing questions for run {run + 1}", total=len(test_data), leave=False):
            result = await process_question(run, question, system, provider, temperature)
            run_results.append(result)

            # Save each question's result immediately
            async with aiofiles.open(checkpoint_file, 'a') as f:
                await f.write(f"{result['Run']},{result['Question']},{result['Correct']},{result['Latency']}\n")

        results.extend(run_results)

    return pd.DataFrame(results)

def save_results(results, output_file):
    """Save results to a CSV file."""
    results.to_csv(output_file, index=False)

async def main(test_file, num_runs, provider, temperature):
    os.makedirs('results', exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join('results', f"{provider}_{os.path.splitext(test_file)[0]}_{now}_t{temperature}_checkpoint.csv")
    test_data = pd.read_csv(os.path.join('tests', test_file))
    results = await run_test(test_data, num_runs, provider, temperature, checkpoint_file)
    output_file = os.path.join('results', f"{provider}_{os.path.splitext(test_file)[0]}_{now}_t{temperature}_final.csv")
    save_results(results, output_file)

if __name__ == "__main__":
    # Example command: python3 main.py CDRE.csv --num_runs 10 --provider openai
    # This runs the CDRE.csv exam questions through openai's flagship LLM 10 times
    parser = argparse.ArgumentParser(description="Run exam benchmark tests")
    parser.add_argument("test_file", help="Name of the test file in the tests folder")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the test (default: 10)")
    parser.add_argument("--provider", type=str, default='openai', 
                       help="The LLM provider to use for this test: 'openai', 'anthropic', 'google', 'llama', 'deepseek', or 'grok' (default: 'openai')")
    parser.add_argument("--temperature", type=str, default=0, help="The temperature to use for this test.")
    args = parser.parse_args()

    asyncio.run(main(args.test_file, args.num_runs, args.provider, args.temperature))
