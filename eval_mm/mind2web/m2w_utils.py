import re
from PIL import Image
import logging
import ast

from prompt_tag import TYPENAME, TYPE_VALUE_NAME, INTERESTED_KEY

logging.basicConfig(level=logging.INFO)

AVAILABLE_ACTION_TYPE=['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

def action2id(action_type):
    if 'CLICK' in action_type.upper() or 'HOVER' in action_type.upper() or 'ENTER' in action_type.upper():
        return 4
    elif 'SELECT' in action_type.upper():
        return 2
    elif 'TYPE' in action_type.upper() or TYPENAME.upper() in action_type.upper():
        return 3
    else:
        return -1
    
def action2step(action):
    action_type = action["operation"]["original_op"]
    assert action_type in AVAILABLE_ACTION_TYPE

    if len(action['pos_candidates'])>0:
        choice = action['pos_candidates'][0]['choice']
        choice_parse = parse_choice(choice)
    else:
        choice_parse = ""

    action_step_dict = dict()
    action_step_dict['action_type'] = action_type.lower()
    action_step_dict[INTERESTED_KEY] = choice_parse
    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        history_action_name = action_type.lower()
        action_step = "{{\"action_type\": \"{}\", \"{}\": \"{}\"}}".format(history_action_name, INTERESTED_KEY, choice_parse)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        history_action_name = action_type.lower()
        action_step = "{{\"action_type\": \"{}\", \"{}\": \"{}\"}}".format(history_action_name, INTERESTED_KEY, select_value)
        action_step_dict[INTERESTED_KEY] = select_value
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": \"{}\", \"{}\": \"{}\", \"{}\": \"{}\"}}".format(TYPENAME, INTERESTED_KEY, choice_parse, TYPE_VALUE_NAME, typed_text)
        action_step_dict['action_type'] = TYPENAME
        action_step_dict[TYPE_VALUE_NAME] = typed_text

    return action_step.replace('"', "'"), action_step_dict

def parse_choice(choice):
    # Remove HTML tags and their contents
    no_tags = re.sub(r'\(([a-z0-9]+)(?:\s+[^)]*?)?\s*', '', choice)
    
    # Remove id attributes
    no_ids = re.sub(r'\s*id=\d+', '', no_tags)
    
    # Remove any remaining parentheses
    no_parens = re.sub(r'[()]', '', no_ids)
    
    # Remove leading/trailing whitespace and collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', no_parens).strip()
    
    # Remove text xxx please select
    cleaned = cleaned.replace('please select', '').strip()
    cleaned = cleaned.replace('text ', '').strip()
    cleaned = cleaned.replace('combobox search q ', '').strip()
    cleaned = cleaned.replace('combobox ', '').strip()
    cleaned = cleaned.replace('option ', '').strip()
    cleaned = cleaned.replace('button ', '').strip()
    cleaned = cleaned.replace('checkbox on ', '').strip()
    cleaned = cleaned.replace("img", "").strip()

    return cleaned.replace('"', "'")

def resize_image(original_image, original_size, max_size = 1344):
    W, H = original_size
    if W<=max_size and H<=max_size:
        return original_image
    if W>=H:
        W_new = max_size
        H_new = int(H/W * W_new)
    else:
        H_new = max_size
        W_new = int(W/H * H_new)
    return original_image.resize([W_new, H_new], Image.Resampling.BICUBIC)

def transform_gt_bbox(original_img_size, new_img_size, bbox):
    w_ratio = new_img_size[0]/original_img_size[0]
    h_ratio = new_img_size[1]/original_img_size[1]
    if bbox['width'] == 0 or bbox['height'] == 0:
        if bbox['width'] == 0 and bbox['height'] == 0:
            bbox['width'] = 50
            bbox['height'] = 50
        elif bbox['height'] == 0:
            bbox['height'] = bbox['width']
        else:
            bbox['width'] = bbox['height']
        bbox['x'] = max(0, bbox['x']-bbox['width']/2)
        bbox['y'] = max(0, bbox['y']-bbox['height']/2)
    bbox['x'] *= w_ratio
    bbox['y'] *= h_ratio
    bbox['width'] *= w_ratio
    bbox['height'] *= h_ratio
    return [bbox['x'], bbox['y'], bbox['x']+bbox['width'], bbox['y']+bbox['height']]


# parse response
def check_available(json_string):
    json_string_eval = json_string.replace('\n', ' ')
    if "'s " in json_string_eval:
        json_string_eval = json_string_eval.replace("'s ", "@@@s ")
        json_string_eval = json_string_eval.replace("'t ", "@@@t ")
    try:
        data = ast.literal_eval(json_string_eval)
        if 'action_type' in data:
            action_type = data['action_type']
            if action_type.lower() in ['click', TYPENAME, 'select', 'enter', 'hover']:
                return True
    except:
        pass
    return False

def coarse_find_action_type(json_string):
    action_pattern = r'[\"\']*(?:action_type|action type|Action type)[\"\']*\s*:\s*(.*?)(?=\s*[,})\n]|$)'
    action_match = re.search(action_pattern, json_string, re.DOTALL)
    if action_match:
        action_type = action_match.group(1).strip()
    else:
        if f"'{TYPENAME}'" in json_string:
            action_type = TYPENAME
        elif "'select'" in json_string:
            action_type = 'select'
        else:
            action_type = 'click'
    return action_type

def coarse_find_value(json_string):
    # parse input value
    # value_pattern = r'"value"\s*:\s*(.*?)(?=\s*[,}]|$)'
    value_pattern = r'[\"\'\s]*(?:input_value|value)[\"\'\s]*:\s*(.*?)(?=\s*[,})]|$)'
    value_match = re.search(value_pattern, json_string, re.DOTALL)
    if value_match:
        value = value_match.group(1).strip()
    else:
        lan_pattern = r'[^\'"](?:input|Input)\s*[\'"]([^\'"]+)[\'"]'
        value_match = re.search(lan_pattern, json_string, re.DOTALL)
        if value_match:
            value = value_match.group(1).strip()
        else:
            value = ""
            logging.error(f"Fail to parse value: {json_string}")
    return value

def coarse_find(data):
    if INTERESTED_KEY in data:
        return INTERESTED_KEY, data[INTERESTED_KEY]
    
    interested_keys = INTERESTED_KEY.split('_')
    for key in list(data.keys()):
        if 'action' in key or 'value' in key:
            continue
        # LLM may not precisely follow instructed key
        for _key in key.split('_'):
            if _key in interested_keys:
                return key, data[key]
    return INTERESTED_KEY, None

def coarse_find_interested_key(json_string):
    coarse_interested_keys = f'{INTERESTED_KEY}|element|description'
    pattern = fr'[\"\'\s]*(?:{coarse_interested_keys})["\'\\s]*:\s*(.*?)(?=\s*[,}})\n]|$)'
    match = re.search(pattern, json_string, re.DOTALL)
    
    if match:
        element_content = match.group(1).strip()
        # Get the start index of the matched content in the original string
        start_index = match.start(1)
    else:
        # Naive select the last 10 characters
        element_content = json_string[-30:]
        start_index = len(json_string)-30
        print('fail to parse from json_string')
    return element_content, start_index

def find_element_content_indices(json_string, tokenizer):
    """Parse the JSON string"""
    from_flag='except'
    try:
        # In case there are multiple action steps in the response, we directly select the last one
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, json_string)
        while len(matches)>1 and not check_available(matches[-1]):
            # if more than 1 actions are generated, filtering out actions with not available action_type
            matches = matches[:-1]
        if len(matches)>=1:
            json_string_new = matches[-1]
        else:
            json_string_new = json_string
        
        # Get the value of element_content_or_description
        json_string_eval = json_string_new.replace('\n', ' ')
        tmp_switch = False
        if "'s " in json_string_eval or "'t " in json_string_eval:
            tmp_switch = True
            json_string_eval = json_string_eval.replace("'s ", "@@@s ") # fix bug: "'s "
            json_string_eval = json_string_eval.replace("'t ", "@@@t ") # fix bug: "won't"
        data = ast.literal_eval(json_string_eval)
        interested_key_parse, element_content = coarse_find(data)
        if tmp_switch: 
            element_content = element_content.replace("@@@s ", "'s ")
            element_content = element_content.replace("@@@t ", "'t ")
        if interested_key_parse != INTERESTED_KEY:
            # switch the interested key
            data.pop(interested_key_parse)
        data[INTERESTED_KEY] = element_content
        
        if element_content is None:
            raise LookupError('element_content is None!')
        # Find the start index of the value
        start_index = json_string.index(element_content)
        from_flag='try'
    except:
        print(">"*100+ f"Fail to parse: {json_string}\n\n")
        data = dict()
        data['action_type'] = coarse_find_action_type(json_string)
        element_content, start_index = coarse_find_interested_key(json_string)
        data[INTERESTED_KEY] = element_content
        if data['action_type'] == TYPENAME:
            data[TYPE_VALUE_NAME] = coarse_find_value(json_string)
        
    # Find the end index of the value
    json_tokens = tokenizer.encode(json_string)
    start_token_index = len(tokenizer.encode(json_string[:start_index]))
    end_token_index = start_token_index + len(tokenizer.encode(element_content))
    try:
        assert tokenizer.decode(json_tokens[start_token_index:end_token_index]) == element_content, f'decode:{tokenizer.decode(json_tokens[start_token_index:end_token_index])}, parse:{element_content}'
    except:
        start_token_index -= 1
        end_token_index -= 1
        end_token_index = min(end_token_index, len(json_tokens)-1)
        print("*"*100+f"\n{from_flag}: decode:{tokenizer.decode(json_tokens[start_token_index:end_token_index])} -> parse:{element_content}\n"+"*"*100)
    return list(range(start_token_index, end_token_index)), data


# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1