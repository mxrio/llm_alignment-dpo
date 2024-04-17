# Imports
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
    # amount_samples = 55                    # for testing purposes
    amount_samples = pref_data_labeled_sample.shape[0]
    # annotation_llm = 'llama2'                   # possible others: llama2:13b-chat, llama2:70b-chat
    # annotation_llm = 'llama2:13b-chat'          # possible others: llama2:13b-chat, llama2:70b-chat
    annotation_llm = 'llama2'                     # possible others: llama2:13b-chat, llama2:70b-chat


    # Function to get annotation from the LLM
    # def get_annotation(dataset_eval, dataset_examples, model, position=0, reversed_order=False, ip_adress='10.1.25.122'):
    def get_annotation(dataset_eval, dataset_examples, model, position=0, reversed_order=False, ip_adress='localhost'):
        # print('starting annotation function')
        # print(f'Dataset eval: {dataset_eval.shape} \t Dataset examples: {dataset_examples.shape} \t Model: {model}')
        base_preamble = 'You are an expert summary rater. Given a piece of text and two of its possible summaries, output only 1 or 2 to indicate which summary is better. Return only "1" or "2" without an explanation! Stop generating text after you have made your choice and returned a number. \n\n'
        detailed_preable = 'A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality. Coherence: This axis answers the question “how coherent is the summary on its own?” A summary is coherent if it’s easy to understand when read on its own and free of English errors. A summary is not coherent if it’s difficult to understand what the summary is trying to say. Generally, it’s more important that the summary is understandable than it being free of grammar errors. Accuracy: This axis answers the question “does the factual information in the summary accurately match the post?” A summary is accurate if it doesn’t say things that aren’t in the article, it doesn’t mix up people, and generally is not misleading. Coverage: This axis answers the question “how well does the summary cover the important information in the post?” A summary has good coverage if it mentions the main information from the post that’s important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice). Overall quality: This axis answers the question “how good is the summary overall at representing the post?” This can encompass all of the above axes of quality, as well as others you feel are important. If it’s hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad. You are an expert summary rater. Given a piece of text and two of its possible summaries, output only 1 or 2 to indicate which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above.'
        instructions = 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation! \n\n'
        if reversed_order:
            sample_to_annotate =   'Title: {title}\n\
                                    Text: {post}\n\
                                    Summary 1: {summary0}\n\
                                    Summary 2: {summary1}\n\n\
                                    Preferred Summary='\
                                    .format(post=dataset_eval['post'][position], title=dataset_eval['title'][position], summary0=dataset_eval['summary1'][position], summary1=dataset_eval['summary0'][position])
        else:
            sample_to_annotate =   'Title: {title}\n\
                                    Text: {post}\n\
                                    Summary 1: {summary0}\n\
                                    Summary 2: {summary1}\n\n\
                                    Preferred Summary='\
                                    .format(post=dataset_eval['post'][position], title=dataset_eval['title'][position], summary0=dataset_eval['summary0'][position], summary1=dataset_eval['summary1'][position])

        chat_content = [
            # System
            {
                'role': 'system',
                'content': base_preamble + '\n\n' + detailed_preable 
            },
            # Example 1
            {
                'role': 'user',
                'content': 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation! \n\n\
                            Title: {title}\n\
                            Text: {post}\n\
                            Summary 1: {summary0}\n\
                            Summary 2: {summary1}\n\
                            Preferred Summary='\
                            .format(post=dataset_examples['post'][0], title=dataset_examples['title'][0], summary0=dataset_examples['summary0'][0], summary1=dataset_examples['summary1'][0])                    
            },
            {
                'role': 'assistant',
                'content': '2'
            },
            # Example 2
            {
                'role': 'user',
                'content': 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation! \n\n\
                            Title: {title}\n\
                            Text: {post}\n\
                            Summary 1: {summary0}\n\
                            Summary 2: {summary1}\n\
                            Preferred Summary='\
                            .format(post=dataset_examples['post'][2], title=dataset_examples['title'][2], summary0=dataset_examples['summary0'][2], summary1=dataset_examples['summary1'][2])                    
            },
            {
                'role': 'assistant',
                'content': '1'
            },
            # Example 3
            {
                'role': 'user',
                'content': 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation! \n\n\
                            Title: {title}\n\
                            Text: {post}\n\
                            Summary 1: {summary0}\n\
                            Summary 2: {summary1}\n\
                            Preferred Summary='\
                            .format(post=dataset_examples['post'][4], title=dataset_examples['title'][4], summary0=dataset_examples['summary0'][4], summary1=dataset_examples['summary1'][4])                    
            },
            {
                'role': 'assistant',
                'content': '2'
            },
            # Example 4
            {
                'role': 'user',
                'content': 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation! \n\n\
                            Title: {title}\n\
                            Text: {post}\n\
                            Summary 1: {summary0}\n\
                            Summary 2: {summary1}\n\
                            Preferred Summary='\
                            .format(post=dataset_examples['post'][6], title=dataset_examples['title'][6], summary0=dataset_examples['summary0'][6], summary1=dataset_examples['summary1'][6])                    
            },
            {
                'role': 'assistant',
                'content': '1'
            },
            # Example 5
            {
                'role': 'user',
                'content': 'Which of the following summaries is better? Consider the coherence, accuracy, coverage, and overall quality of each summary. Return only "1" or "2" without an explanation! \n\n\
                            Title: {title}\n\
                            Text: {post}\n\
                            Summary 1: {summary0}\n\
                            Summary 2: {summary1}\n\
                            Preferred Summary='\
                            .format(post=dataset_examples['post'][9], title=dataset_examples['title'][9], summary0=dataset_examples['summary0'][9], summary1=dataset_examples['summary1'][9])                    
            },
            {
                'role': 'assistant',
                'content': '2'
            },

            # Final Prompt
            {
                'role': 'user',
                'content': instructions + sample_to_annotate
            }
        ]
        payload = {
        'model': model,
        'messages': chat_content,
        'stream': False
    }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'http://{ip_adress}:11434/api/chat', data=json.dumps(payload), headers=headers)
    
        return response


    # Create Dataframe to store the LLM feedback
    # ai_feedback = pd.DataFrame(columns=['human', 'ai', 'ai_reversed', 'duration_ai', 'duration_ai_reversed', 'response_ai', 'response_ai_reversed'])
    # ai_feedback['human'] = (pref_data_labeled_sample['preference']+1).copy()
    # ai_feedback = ai_feedback.reset_index(drop=True)
    ai_feedback = pd.read_feather('data/ai_feedback-llama2-2024-04-16-v2.feather')


    # Start timer
    start_time = time.time()
    current_date = time.strftime("%Y-%m-%d")

    annotation_checkpoints = 25
    amount_samples = 53125

    for sample in range(52450, amount_samples):
        if sample % annotation_checkpoints == 0:
            ai_feedback.to_feather(f'data/ai_feedback-{annotation_llm}-{current_date}-v3.feather')
            print('Annotation data saved')

        annotation_ai = get_annotation(pref_data_labeled_sample, pref_data_labeled_training, annotation_llm, sample).json()
        annotation_ai_reversed = get_annotation(pref_data_labeled_sample, pref_data_labeled_training, annotation_llm, sample, reversed_order=True).json()

        ai_feedback.loc[sample, 'ai'] = annotation_ai['message']['content']
        ai_feedback.loc[sample, 'ai_reversed'] = annotation_ai_reversed['message']['content']
        # ai_feedback.loc[i, 'duration_ai'] = round(annotation_ai['total_duration']/(1000000000), 2)
        # ai_feedback.loc[i, 'duration_ai_reversed'] = round(annotation_ai_reversed['total_duration']/(1000000000), 2)
        ai_feedback.loc[sample, 'response_ai'] = str(annotation_ai)
        ai_feedback.loc[sample, 'response_ai_reversed'] = str(annotation_ai_reversed)
        
        # Estimate remaining time
        remaining_iterations = amount_samples - sample - 1
        current_duration = time.time() - start_time
        average_duration = round(current_duration / (sample+1-52450) / 60, 2) 
        # average_duration = round(((ai_feedback['duration_ai'].mean() + ai_feedback['duration_ai_reversed'].mean()) / 2)/60, 2)
        estimated_time_left = remaining_iterations * average_duration

        print(sample+1,'/', amount_samples, 'samples annotated. \t Estimated time left:', estimated_time_left, 'minutes')

    ai_feedback.to_feather(f'data/ai_feedback-{annotation_llm}-{current_date}-v3.feather')



if __name__ == "__main__":
    # start measurement
    # measurement_process = subprocess.Popen(["python", "measuring_usage.py"])

    # time.sleep(10)
    # print('Annotation started')
    main()
    # print('Annotation done')
    # time.sleep(10)
    
    # stop measurement
    # measurement_process.terminate()

    print('All done!')



