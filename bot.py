from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import re

# Load pre-trained models and tokenizers
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_logic = AutoModelForCausalLM.from_pretrained(model_name)
model_creativity = AutoModelForCausalLM.from_pretrained(model_name)

chaotic_model_name = "gpt2"
chaotic_tokenizer = AutoTokenizer.from_pretrained(chaotic_model_name)
chaotic_model = AutoModelForCausalLM.from_pretrained(chaotic_model_name)

conversation_history_ids_logic = None
conversation_history_ids_creativity = None
show_combined_response = True
super_random_mode = False

# Load random scenarios
def load_super_random_scenarios(file_path):
    try:
        with open(file_path, 'r') as file:
            scenarios = file.readlines()
        return [scenario.strip() for scenario in scenarios]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Using default scenarios.")
        return ["A spaceship lands in a cornfield.", "A cat learns to speak human language."]

random_scenarios = load_super_random_scenarios('super.txt')

def evaluate_math_expression(expression):
    try:
        expression = expression.lower().replace('times', '*').replace('multiplied by', '*').replace('divided by', '/').replace('plus', '+').replace('minus', '-')
        expression = re.sub(r'[^\d\+\-\*/\.\(\)]', '', expression)
        result = eval(expression)
        return f"The result is {result}."
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return "I'm having trouble solving that math problem."
def generate_essay(topic):
    introduction_prompt = f"Write an introduction about {topic}."
    body_prompt = f"Write detailed paragraphs about {topic}."
    conclusion_prompt = f"Write a conclusion about {topic}."

    def generate_text(prompt, max_length):
        input_ids = chaotic_tokenizer.encode(prompt + chaotic_tokenizer.eos_token, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        output = chaotic_model.generate(input_ids, max_length=max_length, pad_token_id=chaotic_tokenizer.eos_token_id, attention_mask=attention_mask, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9, do_sample=True)
        return chaotic_tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    introduction_text = generate_text(introduction_prompt, max_length=150)
    body_text = generate_text(body_prompt, max_length=300)
    conclusion_text = generate_text(conclusion_prompt, max_length=150)

    essay = f"Introduction:\n{introduction_text}\n\nBody:\n{body_text}\n\nConclusion:\n{conclusion_text}"
    return essay

def generate_story(prompt):
    input_ids = chaotic_tokenizer.encode(prompt + chaotic_tokenizer.eos_token, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    output = chaotic_model.generate(input_ids, max_length=250, pad_token_id=chaotic_tokenizer.eos_token_id, attention_mask=attention_mask, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9, do_sample=True)
    story = chaotic_tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return story
def generate_response(tokenizer, model, user_input, conversation_history_ids, personality):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if conversation_history_ids is None:
        personality_prompt = "You are a logical, realistic assistant. Provide clear and factual answers." if personality == "logic" else "You are a creative, imaginative assistant. Provide imaginative and relevant answers."
        personality_input_ids = tokenizer.encode(personality_prompt + tokenizer.eos_token, return_tensors='pt')
        conversation_history_ids = torch.cat([personality_input_ids, new_user_input_ids], dim=-1)
    else:
        conversation_history_ids = torch.cat([conversation_history_ids, new_user_input_ids], dim=-1)

    attention_mask = torch.ones(conversation_history_ids.shape, dtype=torch.long)
    bot_output = model.generate(conversation_history_ids, max_length=150, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    bot_response = tokenizer.decode(bot_output[:, conversation_history_ids.shape[-1]:][0], skip_special_tokens=True)

    special_responses = {
        "topic": {"logic": "Write about the benefits of a balanced diet.", "creativity": "Write about a magical world where animals can talk and have their own societies."},
        "quiet and productive": {"logic": "I choose the quiet and productive town because it is efficient and conducive to achieving goals.", "creativity": "I choose the loud and colorful town because it is vibrant and full of life."},
        "2+2=rabbit": {"logic": "2+2=4 because it is mathematically correct.", "creativity": "2+2=rabbit because why not let our imaginations run wild?"},
        "cats or dogs": {"logic": "I prefer cats because they are independent and low-maintenance.", "creativity": "I love dogs because they are playful and loyal."},
    }

    for keyword, responses in special_responses.items():
        if keyword in user_input.lower():
            return responses.get(personality, bot_response), conversation_history_ids

    if "solve" in user_input.lower() or "help with math" in user_input.lower() or any(op in user_input for op in ["times", "divided by", "multiplied by", "plus", "minus", "+", "-", "*", "/", "**"]):
        expression = user_input.replace("solve", "").replace("help with math", "").strip()
        return evaluate_math_expression(expression), conversation_history_ids
    if "essay" in user_input.lower():
        topic = user_input.split('essay about')[-1].strip()
        return generate_essay(topic), conversation_history_ids

    if personality == "creativity" and "topic" in user_input.lower() and super_random_mode:
        return random.choice(random_scenarios), conversation_history_ids

    return bot_response, conversation_history_ids

lang_mode = False
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    elif user_input.lower() == "switch":
        show_combined_response = not show_combined_response
        mode = "combined" if show_combined_response else "separate"
        print(f"Switched to {mode} mode.")
        continue
    elif user_input.lower() == "super random":
        super_random_mode = not super_random_mode
        mode = "super random" if super_random_mode else "normal"
        print(f"Switched to {mode} mode.")
        continue
    elif user_input.lower() == "set_chaos":
        lang_mode = not lang_mode
        mode = "Chaotic Bot" if lang_mode else "Logic and Creative Bots"
        print(f"Switched to {mode} mode.")
        continue

    if lang_mode:
        prompt = user_input.split('set_chaos')[-1].strip()
        story = generate_story(prompt)
        print(f"Chaotic Bot:\n{story}")
    else:
        logic_response, conversation_history_ids_logic = generate_response(tokenizer, model_logic, user_input, conversation_history_ids_logic, "logic")
        creative_response, conversation_history_ids_creativity = generate_response(tokenizer, model_creativity, user_input, conversation_history_ids_creativity, "creativity")

        if show_combined_response:
            combined_response = logic_response if logic_response == creative_response else f"{logic_response.strip()}, {creative_response.strip()}!"
            print(f"Combined Response:\n{combined_response}")
        else:
            print(f"Logic Bot:\n{logic_response}")
            print(f"Creative Bot:\n{creative_response}")
