import re
import json

def extract_requests_json(llm_response):
  """Extracts the 'requests' JSON from the LLM response using regex."""
  match = re.search(r'"requests":\s*(\[.*?\])', llm_response, re.DOTALL)
  if match:
    requests_json_str = match.group(1)
    try:
      requests_json = json.loads(requests_json_str)
      return requests_json
    except json.JSONDecodeError:
      print("Error: Invalid JSON format for 'requests'")
      return []
  else:
    print("Error: 'requests' key not found in LLM response")
    return []
	
def extract_json_from_string(text):
    match = re.search(r'\{(.*?)\}', text, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return None  # Handle invalid JSON
    return None  # No match found