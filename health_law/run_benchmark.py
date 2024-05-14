import pandas as pd
import re
import json
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from anthropic import Anthropic

from datetime import datetime
import os
import argparse

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


client = OpenAI(api_key = OPENAI_API_KEY, organization = ORGANIZATION_ID)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


path_to_question_dataset = "health_law/datasets/questions/health_law_questions_gpt4.csv"
path_to_ethics_prompts = 'prompting/ethics_prompts.json'
path_to_formating_prompts = 'prompting/formating_prompts.json'

ethics_prompts = json.load(open(path_to_ethics_prompts))
formating_prompts = json.load(open(path_to_formating_prompts))


#*  Read and clearn
question_dataset = pd.read_csv(path_to_question_dataset)
question_dataset = question_dataset.rename(columns={"questions_gpt-4.1": "questions_gpt-4"})
question_dataset = question_dataset.dropna()
question_dataset['gold_answers_gpt-4'] = question_dataset['gold_answers_gpt-4'].apply(lambda x: ''.join([ y for y in x if y.isalnum()])[0])
question_dataset = question_dataset[question_dataset['gold_answers_gpt-4'].isin(['A','B','C','D'])]

# print(question_dataset.head(5))

mach_ethics_prompt= ethics_prompts["machiavelli_prompts"]["MACHIAVELLI_ETHICS_PROMPT"]["prompt"]
json_formating_prompt = formating_prompts["json"]["instruction"] + formating_prompts["json"]["health_law_example_json_string"]

# Specify experiment parameters
# models = ["open-mixtral-8x22b",  "claude-3-haiku-20240307", "claude-3-opus-20240229", "gpt-3.5-turbo", "gpt-4"]
models = ["claude-3-opus-20240229"]

ethics_prompts_names = ["no_prompt", "utilitarian_prompts", "hippocratic_prompt"]


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run triage experiments with specified conditions")
parser.add_argument("--model", nargs="*", choices=models, help="Specify model(s) to use")
parser.add_argument("--ethics_prompts", nargs="*", choices=ethics_prompts_names, help="Specify model to use")
args = parser.parse_args()

# Adjust conditions based on user input
if args.model:
    models = args.model
if args.ethics_prompts:
    ethics_prompts_names = args.ethics_prompts



for model in models: 
  print(model)
  print()
  for ethics_prompt_name in ethics_prompts_names:
    ethics_prompt = ""
    if ethics_prompt_name != "no_prompt":
      ethics_prompt = ethics_prompts[ethics_prompt_name]["prompt"] + " ".join(ethics_prompts[ethics_prompt_name]["steps"])
    print(ethics_prompt_name)
    print()
    question_dataset[f'model_reasoning_{model}_{ethics_prompt_name}'] = None
    question_dataset[f'model_answer_{model}_{ethics_prompt_name}'] = None


    for index, row in question_dataset.iterrows():
        question =  ethics_prompt + question_dataset.at[index, 'questions_gpt-4'] + question_dataset.at[index, 'answer_options'] +  json_formating_prompt
        context= ""
        print("asking...")
        print(question)

        args = {
          "model": model,
          "messages": [
                    {"role": "system", "content": context },
                    {"role": "user", "content": question }
                  ],
          "temperature": 0.5,
        }
        if model == "gpt-3.5-turbo": args["response_format"] = response_format= { "type": "json_object" }

        try:
          if model == "open-mixtral-8x22b":
            completion = mistral_client.chat(**args)
            completion = completion.choices[0].message.content

          elif model == "claude-3-haiku-20240307" or model == "claude-3-opus-20240229":
            completion = anthropic_client.beta.tools.messages.create(
                  model="claude-3-haiku-20240307",
                  max_tokens=1024,
                  temperature= 0.25, #ANTRHROPIC TEMPERATURE SCALE IS FROM 0-1 
                  system = context,
                  messages=[
                        {"role": "user", "content": question }
                    ],
                  tools=[
                    {
                        "name": "triage_classification",
                        "description": "A JSON object containing decision and reasoning about which answer is the right one and why.",
                        "input_schema": {
                          "type": "object",
                          "properties": {
                            "reasoning": {
                              "type": "string",
                              "description": "The reasoning behind the decision."
                            },
                            "action": {
                              "type": "string",
                              "description": "The answer 'A', or 'B'."
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

            completion = client.chat.completions.create(**args)
            completion = completion.choices[0].message.content
          # Assign the computed value to the new column
          model_answer = json.loads(completion)
          print(model_answer)
          reasoning = model_answer["reasoning"]
          answer = model_answer["action"]
          print()
          print(reasoning)
          print(answer)

          # if question_dataset[f'model_anwer_{model}_{ethics_prompt_name}'].dtype != 'string':
          #   question_dataset[f'model_anwer_{model}_{ethics_prompt_name}'] = question_dataset[f'model_anwer_{model}_{ethics_prompt_name}'].astype('string')

          question_dataset.at[index, f'model_reasoning_{model}_{ethics_prompt_name}'] = reasoning
          question_dataset.at[index, f'model_answer_{model}_{ethics_prompt_name}'] = answer
          
          question_dataset.to_csv('health_law/datasets/results/health_results_temp.csv')

          print("answered")
          # break

        except:
          print("an exception occured")
          # break
          continue
  #   break
  # break

# Get date and time
now = datetime.now().strftime("%Y-%m-%d_%H_%M")

question_dataset.to_csv(f'health_law/datasets/results/health_law_answers_claude{now}.csv')
question_dataset


if __name__ == "__main__":
    # Additional logic for command-line interaction, if necessary
    print("Experiment finished. Results saved.")