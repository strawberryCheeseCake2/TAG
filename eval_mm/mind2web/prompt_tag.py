TYPENAME = 'input'
TYPE_VALUE_NAME = 'input_value'
INTERESTED_KEY = "element_description"

history_template="""{previous_actions}"""

action_type_str = 'click, input'
action_space="""Use ONLY the following action types: {action_types}.""".format(action_types=action_type_str)

desc_prompt="exact text in the input field if present, otherwise output up to three keywords"

type_statement="""For input actions: {{'action_type': '{type_name}', '{interested_key}': '[{desc_prompt}]', '{value}': '[content to be entered into the input field]'}}""".format(interested_key=INTERESTED_KEY, type_name=TYPENAME, value=TYPE_VALUE_NAME, desc_prompt=desc_prompt)

query_prompt = """Your task is to achieve the goal: "{goal}"
# Context
- The given image is the latest observed UI screenshot.
- Some actions may have been performed to reach the latest UI and they are listed in the 'Performed Actions' section for your reference.

#Instructions
- Based on the given UI screenshot, determine the next best action to progress towards the goal.
- {action_space}
- Output EXACTLY ONE action in JSON format with the following structure:
For click actions: {{'action_type': 'click', 'element_description': '[exact text on the element if present, otherwise output up to three keywords]'}}
{type_statement}
- Your output should ONLY contain this JSON object, nothing else.
- DO NOT attempt to actually interact with the UI. Your task is to output the next action for the user to perform.

#Performed Actions
{history}
DO NOT repeat any action listed above. Next step: """