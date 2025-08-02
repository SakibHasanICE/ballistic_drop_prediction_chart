# check_status.py
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

job_id = "ftjob-FmhdhsNeLPFkRvmWDYwDohU4"  # your job ID

job = openai.fine_tuning.jobs.retrieve(job_id)

print(f"Status       : {job.status}")
print(f"Created at   : {job.created_at}")
print(f"Model used   : {job.model}")
print(f"Trained model: {job.fine_tuned_model if job.status == 'succeeded' else 'Not ready yet'}")
# If the job is still running, you can check the status again later.
