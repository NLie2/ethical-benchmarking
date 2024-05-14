import sys
import json
import pandas as pd

from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import Anthropic

from datetime import datetime
import os
from dotenv import load_dotenv
import argparse

# Import key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PRIVATE_ORG_KEY = os.getenv("PRIVATE_ORG_KEY")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY, organization=PRIVATE_ORG_KEY)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# Load files
df = pd.read_csv("triage_experiments/datasets/triage_questions.csv")

triage_zones_and_prompts = json.load(open('triage_experiments/prompting/triage_zones_and_prompts.json'))
triage_scenarios = json.load(open('triage_experiments/prompting/triage_scenarios.json'))
ethics_prompts = json.load(open('prompting/ethics_prompts.json'))
formatting_prompts = json.load(open('prompting/formating_prompts.json'))

personas = json.load(open('context_changes/contexts/prompt_for_nathalie.json'))
personas_names = ["no_prompt", "healthcare_assistant", "doctor_assitant"]


# Set default conditions
start_or_jumpSTART = "start"
# models = ["open-mixtral-8x22b",  "claude-3-haiku-20240307", "claude-3-opus-20240229", "gpt-3.5-turbo", "gpt-4"]
models = ["claude-3-opus-20240229"]
triage_versions = ["from_paper", "action_oriented", "outcome_oriented"]
nr_runs = 1

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run triage experiments with specified conditions")
parser.add_argument("--model", nargs="*", choices=models, help="Specify model(s) to use")
parser.add_argument("--triage_version", nargs="*", choices=triage_versions, help="Specify triage version(s) to use")
parser.add_argument("--nr_runs", type=int, help="Specify how many times you want to run each model per all conditions")
args = parser.parse_args()

# Adjust conditions based on user input
if args.model:
    models = args.model
if args.triage_version:
    triage_versions = args.triage_version
if args.nr_runs:
    nr_runs = args.nr_runs

context = triage_scenarios['contexts'][start_or_jumpSTART] + triage_scenarios['scenarios'][start_or_jumpSTART] # "choose_context"

json_prompt = formatting_prompts['json']['instruction']
example_output = formatting_prompts['json']['triage_example_json_string']

for model in models: 
  for n in range(nr_runs):
    for triage_version in triage_versions:
      for persona_name in personas_names: 
        triage_role = triage_zones_and_prompts[triage_version]['role']  # "choose_role"
        triage_zones = ''.join(triage_zones_and_prompts[triage_version]['zones']) # "choose_triage"
        triage_instruction = triage_zones_and_prompts[triage_version]['prompt'] # "choose_prompt"


        df[f'{model}_{persona_name}_{triage_version}_reasoning_{n}'] = None
        df[f'{model}_{persona_name}_{triage_version}_answer_{n}'] = None

        context = personas['neg'][persona_name] + context + triage_role + triage_zones 
        print(context)

        for index, row in df.iterrows():
          question_prompt = triage_instruction + row['question'] + json_prompt
          print(question_prompt)
          print()
          messages = [
                          {"role": "system", "content": context},
                          {"role": "user", "content": question_prompt }]
          temperature = 0 

          try:
              if model == "open-mixtral-8x22b":
                    completion = mistral_client.chat(
                      model=model,
                      # messages=[ChatMessage(role="user", content="What is the best French cheese?")]
                      messages = messages, 
                      temperature= temperature
                    )
                    completion = completion.choices[0].message.content
              elif model == "claude-3-haiku-20240307" or model == "claude-3-opus-20240229":
                    completion = anthropic_client.beta.tools.messages.create(
                      model=model,
                      max_tokens=1024,
                      system = context,
                      temperature = temperature, 
                      messages=[
                            {"role": "user", "content": question_prompt }
                        ],
                      tools=[
                        {
                            "name": "triage_classification",
                            "description": "A JSON object containing decision and reasoning about which category will be assigned to a person and why.",
                            "input_schema": {
                              "type": "object",
                              "properties": {
                                "reasoning": {
                                  "type": "string",
                                  "description": "The reasoning behind the decision."
                                },
                                "action": {
                                  "type": "string",
                                  "description": "The category 'MINOR', 'EXPECTANT/DECEASED', 'DELAYED', or 'IMMEDIATE'."
                                }
                              }
                            }
                          } 
                        ]
                      ) 
                    print(completion)
                    
                    try: 
                      #haiku
                      completion = json.dumps(completion.content[0].input)
                    except:
                      #opus
                      completion = json.dumps(completion.content[1].input)
                    print(completion)                    

              else: 
                  completion = client.chat.completions.create(
                        model= model,
                        # response_format= { "type": "json_object" },
                        messages= messages,
                        temperature= temperature
                        )
                  completion =  completion.choices[0].message.content

              json_response = json.loads(completion)
              print("answered!", json_response)
              print()


              df.loc[index, f'{model}_{persona_name}_{triage_version}_reasoning_{n}'] = json_response["reasoning"]
              df.loc[index, f'{model}_{persona_name}_{triage_version}_answer_{n}'] = json_response["action"]
              df.to_csv(f'triage_experiments/datasets/results/triage_evil_results.csv')
              # break
            
          except:
              # break
              print("an exception occured")
              continue
  #       break
  #     break
  #   break
  # break

print("Done!")
print(df.head(5))


## Save results in file maked by date and time
# Get date and time
now = datetime.now().strftime("%Y-%m-%d_%H_%M")
df.to_csv(f'triage_experiments/datasets/results/{now}_triage_evil_results_claude_opus.csv')

## Let user choose model, triage_version, and prompt 
## Specify default if no instructions are given 

if __name__ == "__main__":
    # Additional logic for command-line interaction, if necessary
    print("Experiment finished. Results saved.")