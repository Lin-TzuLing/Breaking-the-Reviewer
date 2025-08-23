"""
    This script is used to process the PeerRead/AgentReview dataset.
    We combine all the reviews for one paper and save them in a single file.
    The output file is a txt file with the following format:
        filename: <paper_id>.txt
        content: <review_1>\n<review_2>\n...
    Then we can use the txt file to annotate aspects for each review through ReviewAdvisor repo.
"""


import os
import json
import re

    
def normalize_string(s):
    return re.sub(r'\s+', ' ', s).strip()


def process_PeerRead(dataset_dir, output_dir):
    """
    Process the PeerRead dataset.
    Args:
        dataset_dir: str, the path to the PeerRead dataset.
        output_dir: str, the path to save the processed files.
    """

    # Load the data
    review_text = {}
    num_avg_reviews, num_min_reviews = 0, 100
    for filename in sorted(os.listdir(dataset_dir)):
        paper_id = filename.split('.')[0]
        with open(dataset_dir+'/'+filename, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())  
            reviews = [normalize_string(r['comments']) for r in data['reviews']]
            reviews = [r for r in reviews if r != ''] # filter out empty comments
            review_text[paper_id] = sorted(list(set(reviews)))
            
            if len(review_text[paper_id]) < num_min_reviews:
                num_min_reviews = len(review_text[paper_id])
            num_avg_reviews += len(review_text[paper_id])         
            
    print(f"avg reviews per paper: {(num_avg_reviews/len(review_text)):.4f}")
    print(f"min reviews per paper: {num_min_reviews}")
 
    
    # Save the reviews in the format of sample.txt from the ReviewAdvisor repository
    os.makedirs(output_dir, exist_ok=True)
    for paper_id, reviews in review_text.items():
        with open(os.path.join(output_dir, f'{paper_id}.txt'), 'w') as f:
            cleaned_reviews = [review.replace('\n', '') for review in reviews]
            f.write('\n'.join(cleaned_reviews))
    
    print(f"Processed {len(review_text)} papers. Saved to {output_dir}")
    

def process_AgentReview(dataset_dir, output_dir):
    """
    Process the AgentReview dataset.
    Args:
        dataset_dir: str, the path to the AgentReview dataset.
        output_dir: str, the path to save the processed files.
    """
    
    def rename(target_path: str) -> str:
        """Rename the output to avoid the name confliction."""
        parts = target_path.split('/')
        conference = parts[3]        # ICLR2022
        decision = parts[5]          # Accept-oral
        file_id = os.path.splitext(parts[6])[0]  
        return f"{conference}_{decision}_{file_id}"
    
    # Load the data
    review_key_mapping = {'2020': 'review', '2021':'review',
                              '2022': 'main_review', '2023': 'summary_of_the_review'}
    rating_key_mapping = {'2020': 'rating', '2021':'rating',
                            '2022': 'recommendation', '2023': 'recommendation'}
    
    review_text = {}
    num_avg_reviews, num_min_reviews = 0, 100

    for decision_subset in sorted(os.listdir(dataset_dir)): # ['Reject', 'Accept-oral', 'Accept-poster', 'Accept-spotlight']
        for filename in sorted(os.listdir(f"{dataset_dir}/{decision_subset}")):
            full_path = os.path.join(dataset_dir, decision_subset, filename)
            paper_id = rename(full_path).split('.')[0]
            year = paper_id.split('_')[0][4:]

            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())  
                
                # filter out empty comments and make sure replies are from reviewers not authors
                reviews = [
                            r['content'][review_key_mapping[year]] for r in data['details']['replies']
                            if any(key in r['content'] for key in rating_key_mapping.values()) and 
                                r['content'][review_key_mapping[year]] != ''
                            ]
                reviews = [normalize_string(r) for r in reviews if normalize_string(r) != ''] # filter out empty comments
                review_text[paper_id] = sorted(list(set(reviews)))
                  
                if len(review_text[paper_id]) < num_min_reviews:
                    num_min_reviews = len(review_text[paper_id])
                num_avg_reviews += len(review_text[paper_id])         
            
    print(f"avg reviews per paper: {(num_avg_reviews/len(review_text)):.4f}")
    print(f"min reviews per paper: {num_min_reviews}")
 
    
    # Save the reviews in the format of sample.txt from the ReviewAdvisor repository
    os.makedirs(output_dir, exist_ok=True)
    for paper_id, reviews in review_text.items():
        with open(os.path.join(output_dir, f'{paper_id}.txt'), 'w') as f:
            cleaned_reviews = [review.replace('\n', '') for review in reviews]
            f.write('\n'.join(cleaned_reviews))
    
    print(f"Processed {len(review_text)} papers. Saved to {output_dir}")      
            
          
if __name__ == '__main__':
    import sys
          
    if len(sys.argv) < 4:
        print("Please provide the dataset name (PeerRead or AgentReview) as an argument.")
        print("Usage: python tagger_preprocess.py <dataset_name> <dataset_dir> <output_dir>")
        sys.exit(1)

    DATASET = sys.argv[1]
    DATASET_DIR = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]

    if DATASET == "PeerRead":
        process_PeerRead(DATASET_DIR, OUTPUT_DIR)

    elif DATASET == "AgentReview":
        process_AgentReview(DATASET_DIR, OUTPUT_DIR)

    else:
        print(f"Unknown dataset: {DATASET}")
        print("Please provide either 'PeerRead' or 'AgentReview' as an argument.")
      
