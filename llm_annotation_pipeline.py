import pandas as pd
import requests 
import json
import time
import subprocess

def main():
    # Load the labeled preference data
    pref_data_labeled_sample = pd.read_feather('data/pref_data_labeled-60k_sample.feather')
    pref_data_labeled_training = pd.read_feather('data/pref_data_labeled.feather')[:10]

    # Define annotation parameters
    amount_samples = pref_data_labeled_sample.shape[0]
    annotation_llm = 'llama2'

    # Function to get annotation from the LLM
    def get_annotation(dataset_eval, dataset_examples, model, position=0, reversed_order=False, ip_adress='localhost'):
        base_preamble = 'You are an expert summary rater. Given a piece of text and two of its possible summaries, output only 1 or 2 to indicate which summary is better. Return only "1" or "2" without an explanation! Stop generating text after you have made your choice and returned a number.\n\n'
        detailed_preable = 'Detailed instructions...'
        instructions = 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation!\n\n'
        if reversed_order:
            sample_to_annotate = 'Title: {title}\nText: {post}\nSummary 1: {summary0}\nSummary 2: {summary1}\n\nPreferred Summary='.format(post=dataset_eval['post'][position], title=dataset_eval['title'][position], summary0=dataset_eval['summary1'][position], summary1=dataset_eval['summary0'][position])
        else:
            sample_to_annotate = 'Title: {title}\nText: {post}\nSummary 1: {summary0}\nSummary 2: {summary1}\n\nPreferred Summary='.format(post=dataset_eval['post'][position], title=dataset_eval['title'][position], summary0=dataset_eval['summary0'][position], summary1=dataset_eval['summary1'][position])

        chat_content = [
            {'role': 'system', 'content': base_preamble + '\n\n' + detailed_preable},
            {'role': 'user', 'content': instructions + sample_to_annotate},
            {'role': 'assistant', 'content': '2'}, # Example 1
            {'role': 'user', 'content': instructions + sample_to_annotate},
            {'role': 'assistant', 'content': '1'}, # Example 2
            {'role': 'user', 'content': instructions + sample_to_annotate},
            {'role': 'assistant', 'content': '2'}, # Example 3
            {'role': 'user', 'content': instructions + sample_to_annotate},
            {'role': 'assistant', 'content': '1'}, # Example 4
            {'role': 'user', 'content': instructions + sample_to_annotate},
            {'role': 'assistant', 'content': '2'}, # Example 5
            {'role': 'user', 'content': instructions + sample_to_annotate}
        ]
        payload = {'model': model, 'messages': chat_content, 'stream': False}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'http://{ip_adress}:11434/api/chat', data=json.dumps(payload), headers=headers)
        return response

    # Create Dataframe to store the LLM feedback
    ai_feedback = pd.DataFrame(columns=['human', 'ai', 'ai_reversed', 'duration_ai', 'duration_ai_reversed', 'response_ai', 'response_ai_reversed'])
    ai_feedback['human'] = (pref_data_labeled_sample['preference']+1).copy()
    ai_feedback = ai_feedback.reset_index(drop=True)

    # Start timer
    start_time = time.time()
    current_date = time.strftime("%Y-%m-%d")
    annotation_checkpoints = 25
    last_checkpoint = 0

    for sample in range(last_checkpoint, amount_samples):
        if sample % annotation_checkpoints == 0:
            ai_feedback.to_feather(f'data/ai_feedback-{annotation_llm}-{current_date}-v3.feather')
            print('Annotation data saved')

        annotation_ai = get_annotation(pref_data_labeled_sample, pref_data_labeled_training, annotation_llm, sample).json()
        annotation_ai_reversed = get_annotation(pref_data_labeled_sample, pref_data_labeled_training, annotation_llm, sample, reversed_order=True).json()

        ai_feedback.loc[sample, 'ai'] = annotation_ai['message']['content']
        ai_feedback.loc[sample, 'ai_reversed'] = annotation_ai_reversed['message']['content']
        ai_feedback.loc[sample, 'response_ai'] = str(annotation_ai)
        ai_feedback.loc[sample, 'response_ai_reversed'] = str(annotation_ai_reversed)
        
        # Estimate remaining time
        remaining_iterations = amount_samples - sample - 1
        current_duration = time.time() - start_time
        average_duration = round(current_duration / (sample+1-last_checkpoint) / 60, 2) 
        estimated_time_left = remaining_iterations * average_duration

        print(sample+1,'/', amount_samples, 'samples annotated. \t Estimated time left:', estimated_time_left, 'minutes')
    ai_feedback.to_feather(f'data/ai_feedback-{annotation_llm}-{current_date}.feather')

if __name__ == "__main__":
    measurement_process = subprocess.Popen(["python", "measuring_usage.py"])

    time.sleep(10)
    print('Annotation started')
    main()
    print('Annotation done')
    time.sleep(10)
    
    measurement_process.terminate()
    print('All done!')
