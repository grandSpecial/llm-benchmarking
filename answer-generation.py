# This script is adapted to include checkpointing for resiliency across long runs.

import os
import time
import asyncio
import pandas as pd
from datetime import datetime
from openai import OpenAI
import google.generativeai as g_client
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import aiofiles

load_dotenv()

# API clients
o_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
g_client.configure(api_key=os.getenv("GEMINI_API_KEY"))
grok_api_key = os.getenv("GROK_API_KEY")

# Prompts
prompts = {
    "zero-shot": "You are a registered dietitian responding to a patient recently discharged from ICU. Provide a concise answer in the format: Yes | No — followed by a brief explanation.",
    "chain-of-thought": "You are a registered dietitian helping a patient understand the risks of their dietary question. Think through the context and explain your reasoning briefly before giving a Yes | No answer with explanation."
}

async def query_model(provider, model, system, user, temperature):
    start = time.time()
    try:
        if provider == "openai":
            completion = await asyncio.to_thread(
                o_client.chat.completions.create,
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=512,
                temperature=temperature
            )
            return completion.choices[0].message.content, time.time() - start

        elif provider == "google":
            gen_model = g_client.GenerativeModel(
                model,
                generation_config=GenerationConfig(temperature=temperature),
                system_instruction=system
            )
            response = await asyncio.to_thread(gen_model.generate_content, user)
            return response.text, time.time() - start

        elif provider == "grok":
            headers = {
                "Authorization": f"Bearer {grok_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "model": model,
                "temperature": temperature,
                "max_tokens": 512
            }
            response = await asyncio.to_thread(
                requests.post,
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            answer = response.json()['choices'][0]['message']['content']
            return answer, time.time() - start

    except Exception as e:
        print(f"{provider} ({model}) error: {e}")
        return None, None

async def run_all():
    df = pd.read_csv("tests/chatbot_questions.csv")

    models = {
        "openai": "gpt-4o",
        "google": "gemini-1.5-pro",
        "grok": "grok-2-1212"
    }

    temperatures = [0, 0.7]
    variants = ["zero-shot", "chain-of-thought"]
    runs = 3

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"results/chatbot_checkpoint_{timestamp}.csv"

    header_written = False
    if os.path.exists(checkpoint_path):
        results = pd.read_csv(checkpoint_path).to_dict('records')
    else:
        results = []

    async with aiofiles.open(checkpoint_path, 'w') as f:
        await f.write("scenario_id,question_id,provider,model,variant,temperature,run,latency,answer\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        for variant in variants:
            for provider, model in models.items():
                for temp in temperatures:
                    for run in range(1, runs + 1):
                        answer, latency = await query_model(provider, model, prompts[variant], row["question"], temp)

                        result = {
                            "scenario_id": row["scenario_id"],
                            "question_id": row["question_id"],
                            "provider": provider,
                            "model": model,
                            "variant": variant,
                            "temperature": temp,
                            "run": run,
                            "latency": latency,
                            "answer": answer
                        }

                        results.append(result)

                        async with aiofiles.open(checkpoint_path, 'a') as f:
                            await f.write(
                                f"{result['scenario_id']},{result['question_id']},{result['provider']},{result['model']},{result['variant']},{result['temperature']},{result['run']},{result['latency']},\"{result['answer'].replace(chr(10), ' ').replace(chr(13), ' ') if result['answer'] else ''}\"\n"
                            )

    pd.DataFrame(results).to_csv(f"results/chatbot_llm_benchmark_{timestamp}.csv", index=False)

if __name__ == "__main__":
    asyncio.run(run_all())
