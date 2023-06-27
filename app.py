from bs4 import BeautifulSoup
import requests
import gradio as gr
import google.generativeai as palm
import config

def get_twitlonger_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        post_content = soup.find('p', id='posttext')
        text = post_content.get_text(separator='\n')

        return text.strip()
    except Exception as e:
        return "Could not find Twitlonger text."

def paraphrase_text(url):
    palm.configure(api_key=config.BARD_API_KEY)

    defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0.7,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
    'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":3},{"category":"HARM_CATEGORY_TOXICITY","threshold":3},{"category":"HARM_CATEGORY_VIOLENCE","threshold":3},{"category":"HARM_CATEGORY_SEXUAL","threshold":3},{"category":"HARM_CATEGORY_MEDICAL","threshold":3},{"category":"HARM_CATEGORY_DANGEROUS","threshold":3}],
    }

    text = get_twitlonger_text(url)

    if text == "Could not find Twitlonger text.":
        return [text, text]
    
    if len(text) <= 280: # Why would you write a Twitlonger less than 280 characters!?
        return [text, text]

    og_prompt = """
        Concisely paraphrase this statement in 1-2 sentences.
        Do not assume the writer's gender, their pronouns should be "they/them."

    """
    prompt = og_prompt + f"{text}"

    # Recursively paraphrases until length of tweet < 280
    def generate_response(result, attempts):
        if attempts <= 0:
            return [text, "Failed to generate a concise response. This may be the bot's fault or the content of the Twitlonger is too extreme."]
        
        response = palm.generate_text(
            **defaults,
            prompt=og_prompt + f"{result}"
        )

        result = response.result

        if result is None:
            generate_response(result, -1)
        
        if len(result) <= 280:
            return [text, result]

        return generate_response(result, attempts - 1)
    
    return generate_response(prompt, attempts=5) # If it can't cut down within 5 recursions, it's joever.

with gr.Blocks() as demo:
    with gr.Row():
        url_input = gr.Textbox(label="Enter the Twitlonger URL here: ")
    
    with gr.Row():
        with gr.Column():
            full_output = gr.Textbox(label="Full Output")
        with gr.Column():
            paraphrased_output = gr.Textbox(label="Paraphrased Output", info="Please remember that no paraphrase is as good or as accurate as the original.")

    with gr.Row():
        btn = gr.Button("Paraphrase Twitlonger")

    btn.click(fn=paraphrase_text, inputs=url_input, outputs=[full_output, paraphrased_output])

    gr.Examples(
        examples=[
            "https://www.twitlonger.com/show/n_1soodca",
            "https://www.twitlonger.com/show/n_1srvbca",
            "https://www.twitlonger.com/show/n_1ss1qiv"
        ],
        label="Example TwitLongers from Twitlonger, A Teamfight Tactics Personality, and A Valorant Dev",
        fn=paraphrase_text,
        inputs=url_input,
        outputs=[full_output, paraphrased_output],
        cache_examples=False,
        run_on_click=True,
    )

demo.launch()