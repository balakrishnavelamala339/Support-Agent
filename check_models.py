from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6LRDlJbcVXtmksXOaT2kHE3YFX2OsfdbrBlsP3vTEYjEQ"
)

for model in client.models.list():
    print(model.name)