import streamlit as st
from streamlit_chat import message

@st.cache(allow_output_mutation=True)
def get_pipe():
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tokenizer = AutoTokenizer.from_pretrained("heegyu/kodialogpt-v1")
    model = AutoModelForCausalLM.from_pretrained("heegyu/kodialogpt-v1")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_response(generator, history, max_context: int = 7, bot_id: str = '1'):
    generation_args = dict(
        num_beams=4,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        eos_token_id=375, # \n
        max_new_tokens=64,
        do_sample=True,
        top_k=50,
        early_stopping=True
    )
    context = []
    for i, text in enumerate(history):
        context.append(f"{i % 2} : {text}\n")
    
    if len(context) > max_context:
        context = context[-max_context:]
    context = "".join(context) + f"{bot_id} : "

    # print(f"get_response({context})")

    response = generator(
        context,
        **generation_args
    )[0]["generated_text"]
    response = response[len(context):].split("\n")[0]
    return response

st.title("kodialogpt-v1 demo")

with st.spinner("loading model..."):
    generator = get_pipe()

if 'message_history' not in st.session_state:
    st.session_state.message_history =  []
history = st.session_state.message_history

# print(st.session_state.message_history)
for i, message_ in enumerate(st.session_state.message_history):
    message(message_,is_user=i % 2 == 0) # display all the previous message

# placeholder = st.empty() # placeholder for latest message
input_ = st.text_input("YOU", value="")

if input_ is not None and len(input_) > 0:
    if len(history) <= 1 or history[-2] != input_:
        with st.spinner("대답을 생성중입니다..."):
            st.session_state.message_history.append(input_)
            response = get_response(generator, history)
            st.session_state.message_history.append(response)
            st.experimental_rerun()
