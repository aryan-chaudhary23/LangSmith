import os
import time
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

print("Creating a test run...")

try:
    run = client.create_run(
        name="manual_test",
        run_type="chain",
        inputs={
            "message": "Hello LangSmith"
        },
        project_name="langgraph-demo"
    )

    print("Run created!")
    print("Run ID:", run.id)

    # Give LangSmith time to process it
    time.sleep(3)

    print("\nChecking projects again...")

    projects = list(client.list_projects())

    for project in projects:
        print("-", project.name)

except Exception as e:
    print("\nERROR")
    print(type(e).__name__)
    print(e)