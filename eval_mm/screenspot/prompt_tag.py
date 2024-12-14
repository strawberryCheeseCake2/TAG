import json
import re

interested_key = "element_content_or_description"
query_prompt = """In this UI screenshot, I want to perform the command "{query_text}", which element should I perform? Please output in json format ONLY with the following two keys: action_type, {interested_key}. If the element is an icon and there is no text content on this element, please output up to three keywords for the {interested_key}."""

def find_element_content_indices(json_string, tokenizer, interested_key="element_content_or_description"):
    # Parse the JSON string
    try:
        data = json.loads(json_string)
        
        # Get the value of element_content_or_description
        element_content = data.get(interested_key)
        
        if element_content is None:
            return []
        # Find the start index of the value
        start_index = json_string.index(f'"{interested_key}": "{element_content}"')
        start_index = start_index + len(f'"{interested_key}": "')
    except:
        pattern = r'"element_content_or_description"\s*:\s*(.*?)(?=\s*[,}]|$)'
        match = re.search(pattern, json_string, re.DOTALL)
        
        if match:
            element_content = match.group(1).strip()
            # Get the start and end indices of the matching content in the original string.
            start_index = match.start(1)
        else:
            print('fail to parse from json_string')
            return []

    json_tokens = tokenizer.encode(json_string)
    start_token_index = len(tokenizer.encode(json_string[:start_index]))
    end_token_index = start_token_index + len(tokenizer.encode(element_content))
    try:
        assert tokenizer.decode(json_tokens[start_token_index:end_token_index]) == element_content, f'decode:{tokenizer.decode(json_tokens[start_token_index:end_token_index])}, parse:{element_content}'
    except:
        start_token_index -= 1
        end_token_index -= 1
    return list(range(start_token_index, end_token_index))