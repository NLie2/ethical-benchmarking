# SCALABLE ETHICAL BENCHMARKING 


## Things this program should be able to do: 

1. A user interface that shows a leaderboard of different models based on different benchmark results

2. Collect ethical benchmarking questions + Gold Answers based on text input 

3. Collect LLM answers to a set of questions 
4. Use a set of different contexts per model 

5. Evaluate the correctness of the answers


## DataStorage
- store data in CSV files 

- Question CSV file 
- Answer CSV file per model 
- Combined CSV file with all data 


# File Structure 
|_ethical_med_qa
|_health_law
  |_ question_generation.py: Collect ethical benchmarking questions + Gold Answers based on text input
  |_ resources
    |_ Medical Law in Austria Book (be careful about publishing on GitHub)
|_triage_experiments
  |_ question_generation.py: Convert triage questions to csv file 
  |_ resources
      |_ triage papers (be careful about publishing on GitHub)
|_test.py: test an AI model on a set of questions (INCLUDE FUNCTIONALITY TO CHANGE CONTEXTS)
|_analysis.py: evaluate correctness of responses and output summary 
  



