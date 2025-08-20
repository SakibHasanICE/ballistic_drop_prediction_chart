import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with your API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def reformat_and_fine_tune_model(input_file_path):
    """
    Reformats the dataset, uploads it, and creates a fine-tuning job.
    """
    output_file_path = "reformatted_dataset.jsonl"
    reformatted_lines = []

    # Reformat the dataset
    print(f"Reformatting dataset from {input_file_path}...")
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            assistant_content_str = data['messages'][1]['content']
            
            # The assistant's content is a string representation of a JSON array.
            # We need to parse it and convert it to a simple string.
            assistant_data = json.loads(assistant_content_str)
            
            reformatted_output = ""
            for item in assistant_data:
                reformatted_output += f"{item['range_yd']} yards: {item['drop_in']} inches, "
            
            # Remove trailing comma and space
            reformatted_output = reformatted_output.rstrip(', ')
            
            data['messages'][1]['content'] = reformatted_output
            reformatted_lines.append(json.dumps(data))

    # Write the reformatted data to a new file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in reformatted_lines:
            outfile.write(line + '\n')
    
    print(f"Dataset successfully reformatted and saved to {output_file_path}")

    try:
        # Upload the reformatted training file
        print(f"Uploading file: {output_file_path}")
        with open(output_file_path, "rb") as f:
            upload_response = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = upload_response.id
        print(f"File uploaded successfully with ID: {file_id}")

        # Create a fine-tuning job
        print("Creating fine-tuning job...")
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-3.5-turbo"
        )
        job_id = fine_tune_response.id
        print(f"Fine-tuning job created successfully with ID: {job_id}")

        print("You can check the status of your job with the following command:")
        print(f"openai api fine_tuning.jobs.get -i {job_id}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    dataset_file = "ballistic_dataset_1500.jsonl"
    if not os.path.exists(".env") or not os.getenv("OPENAI_API_KEY"):
        print("Error: The .env file is missing or the OPENAI_API_KEY is not set.")
        print("Please create a .env file and add your API key like this:")
        print("OPENAI_API_KEY='your_api_key_here'")
    else:
        reformat_and_fine_tune_model(dataset_file)