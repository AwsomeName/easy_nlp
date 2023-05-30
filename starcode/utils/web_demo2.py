from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,  TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
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
    tokenizer.pad_token = tokenizer.eos_token
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
    stop_token_ids = [tokenizer.eos_token_id]
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

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
        input = "Human: " + input + "\n\n\nAssistant: "
        with st.empty():
            # for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                            #    temperature=temperature):
                # query, response = history[-1]
                # st.write(response)
            # inputs = tokenizer.encode(input, return_tensors="pt").cuda()
            if do_sample>0:
                stop = StopOnTokens() 
                tokenizer.pad_token = tokenizer.eos_token
                inputs = tokenizer.encode(input, return_tensors="pt").cuda()
                logger.info("prepare done, start gen...")
                generation_kwargs = dict(
                    inputs = inputs, 
                    streamer=streamer, 
                    max_new_tokens=max_length,
                    top_k = 1,
                    top_p = top_p,
                    # num_beams = 1,
                    # num_beam_groups = 1,
                    repetition_penalty = 1.1,
                    # num_return_sequences=1
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    stopping_criteria=StoppingCriteriaList([stop])
                    )
            
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                generated_text = ""
                for new_text in streamer:
                    generated_text += new_text
                    st.write(generated_text)
            else:
                chosen_token = tokenizer(
                    input,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
        
                input_ids = torch.tensor(chosen_token["input_ids"]).cuda()
                attention_mask = torch.tensor(chosen_token['attention_mask']).cuda()
                outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_length,
                        do_sample=True,
                        top_k = 5,
                        top_p=0.95,
                        num_beams = 3,
                        repetition_penalty = 2.0,
                        num_return_sequences=1)
                # outputs = model.generate(inputs)
                response = tokenizer.decode(outputs[0])    
                st.write(response)

    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

do_sample = st.sidebar.slider(
    "do_sample", 0, 1, 1, step=1
)

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 256, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.8, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.01, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
