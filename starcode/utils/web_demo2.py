from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,  TextIteratorStreamer
import streamlit as st
from streamlit_chat import message
from threading import Thread
from transformers.utils import logging
import torch
from transformers.generation.logits_process import LogitsProcessor
# from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

acc_token = "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"

logger = logging.get_logger(__name__)
st.set_page_config(
    page_title="bigcode/starcoder 演示",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    logger.info("get tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/root/data/starcoder", trust_remote_code=True, use_auth_token=acc_token, padding_side="left")
    # tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", trust_remote_code=True, use_auth_token=acc_token)
    # tokenizer = AutoTokenizer.from_pretrained("/root/easy_nlp/trl/model/bloom-560", trust_remote_code=True, use_auth_token=acc_token)
    logger.info("get model...")
    model = AutoModelForCausalLM.from_pretrained("/root/data/starcoder", trust_remote_code=True, use_auth_token=acc_token).half().cuda()
    # model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", trust_remote_code=True, use_auth_token=acc_token).half().cuda()
    # model = AutoModelForCausalLM.from_pretrained("/root/easy_nlp/trl/model/bloom-560", trust_remote_code=True, use_auth_token=acc_token).half().cuda()
    model = model.eval()
    streamer = TextIteratorStreamer(tokenizer)
    logger.info("init done")
    return tokenizer, model, streamer


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

# class InvalidScoreLogitsProcessor(LogitsProcessor):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if torch.isnan(scores).any() or torch.isinf(scores).any():
#             scores.zero_()
#             scores[..., 5] = 5e4
#         return scores



def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model, streamer = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            # for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                            #    temperature=temperature):
                # query, response = history[-1]
                # st.write(response)
            inputs = tokenizer.encode(input, return_tensors="pt").cuda()
            logger.info("prepare done, start gen...")
            generation_kwargs = dict(
                inputs = inputs, 
                streamer=streamer, 
                max_new_tokens=max_length,
                do_sample=True,
                top_k = 5,
                top_p = top_p,
                num_return_sequences=1)
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                st.write(generated_text)
            # generated_text
            # outputs = model.generate(
            #             inputs,
            #             # attention_mask=attention_mask,
            #             max_length=max_length,
            #             do_sample=True,
            #             top_k = 5,
            #             top_p=0.95,
            #             num_return_sequences=1)
            # # outputs = model.generate(inputs)
            # response = tokenizer.decode(outputs[0])    
            # st.write(response)

    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 1024, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
