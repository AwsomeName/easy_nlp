from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from streamlit_chat import message
import torch
import re
from torch import nn
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

acc_token = "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"


st.set_page_config(
    page_title="bigcode/starcoder 演示",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    # tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", trust_remote_code=True, use_auth_token=acc_token)
    # model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", trust_remote_code=True, use_auth_token=acc_token).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained("/root/easy_nlp/trl/model/bloom-560", trust_remote_code=True, use_auth_token=acc_token)
    model = AutoModelForCausalLM.from_pretrained("/root/easy_nlp/trl/model/bloom-560", trust_remote_code=True, use_auth_token=acc_token).half().cuda()
    model = model.eval()
    return tokenizer, model


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores



def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []
        
    # def process_response(self, response):
    #     response = response.strip()
    #     response = response.replace("[[训练时间]]", "2023年")
    #     punkts = [
    #         [",", "，"],
    #         ["!", "！"],
    #         [":", "："],
    #         [";", "；"],
    #         ["\?", "？"],
    #     ]
    #     for item in punkts:
    #         response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
    #         response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    #     return response
    
    # def prepare_inputs_for_generation(
    #         input_ids: torch.LongTensor,
    #         past: Optional[torch.Tensor] = None,
    #         past_key_values: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         **kwargs
    # ) -> dict:
    #     batch_size, seq_length = input_ids.shape
    #     MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
    #     seqs = input_ids.tolist()
    #     mask_positions, use_gmasks = [], []
    #     for seq in seqs:
    #         mask_token = gMASK if gMASK in seq else MASK
    #         use_gmask = mask_token == gMASK
    #         mask_positions.append(seq.index(mask_token))
    #         use_gmasks.append(use_gmask)

    #     # only last token for input_ids if past is not None
    #     if past is not None or past_key_values is not None:
    #         last_token = input_ids[:, -1].unsqueeze(-1)
    #         if attention_mask is not None and attention_mask.dtype == torch.bool:
    #             attention_mask = attention_mask[:, :, -1:]
    #         else:
    #             attention_mask = None
    #         if position_ids is not None:
    #             position_ids = position_ids[..., -1:]
    #         else:
    #             context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
    #             if self.position_encoding_2d:
    #                 position_ids = torch.tensor(
    #                     [[mask_position, seq_length - context_length] for mask_position, context_length in
    #                      zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
    #             else:
    #                 position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
    #                                             device=input_ids.device).unsqueeze(-1)

    #         if past is None:
    #             past = past_key_values
    #         return {
    #             "input_ids": last_token,
    #             "past_key_values": past,
    #             "position_ids": position_ids,
    #             "attention_mask": attention_mask
    #         }
    #     else:
    #         if attention_mask is not None and attention_mask.dtype != torch.bool:
    #             # logger.warning_once(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
    #             attention_mask = None
    #         if attention_mask is None:
    #             attention_mask = self.get_masks(
    #                 input_ids,
    #                 device=input_ids.device
    #             )
    #         if position_ids is None:
    #             position_ids = self.get_position_ids(
    #                 input_ids,
    #                 device=input_ids.device,
    #                 mask_positions=mask_positions,
    #                 use_gmasks=use_gmasks
    #             )

    #         return {
    #             "input_ids": input_ids,
    #             "past_key_values": past,
    #             "position_ids": position_ids,
    #             "attention_mask": attention_mask
    #         }

    # @torch.no_grad()
    # def stream_generate(
    #         model,
    #         input_ids,
    #         generation_config: Optional[GenerationConfig] = None,
    #         logits_processor: Optional[LogitsProcessorList] = None,
    #         stopping_criteria: Optional[StoppingCriteriaList] = None,
    #         prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #         **kwargs,
    # ):
    #     batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

    #     # if generation_config is None:
    #     #     generation_config = self.generation_config
    #     # generation_config = copy.deepcopy(generation_config)
    #     model_kwargs = generation_config.update(**kwargs)
    #     bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

    #     if isinstance(eos_token_id, int):
    #         eos_token_id = [eos_token_id]

    #     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    #     if has_default_max_length and generation_config.max_new_tokens is None:
    #         pass
    #     elif generation_config.max_new_tokens is not None:
    #         generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
    #         if not has_default_max_length:
    #             pass

    #     # if input_ids_seq_length >= generation_config.max_length:
    #     #     input_ids_string = "decoder_input_ids" if config.is_encoder_decoder else "input_ids"
    #     #     pass

    #     # 2. Set generation parameters if not already defined
    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    #     logits_processor = model._get_logits_processor(
    #         generation_config=generation_config,
    #         input_ids_seq_length=input_ids_seq_length,
    #         encoder_input_ids=input_ids,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         logits_processor=logits_processor,
    #     )

    #     stopping_criteria = model._get_stopping_criteria(
    #         generation_config=generation_config, stopping_criteria=stopping_criteria
    #     )
    #     logits_warper = model._get_logits_warper(generation_config)

    #     unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    #     scores = None
    #     while True:
    #         model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)
    #         # forward pass to get next token
    #         outputs = model(
    #             **model_inputs,
    #             return_dict=True,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #         )

    #         next_token_logits = outputs.logits[:, -1, :]

    #         # pre-process distribution
    #         next_token_scores = logits_processor(input_ids, next_token_logits)
    #         next_token_scores = logits_warper(input_ids, next_token_scores)

    #         # sample
    #         probs = nn.functional.softmax(next_token_scores, dim=-1)
    #         if generation_config.do_sample:
    #             next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    #         else:
    #             next_tokens = torch.argmax(probs, dim=-1)

    #         # update generated ids, model inputs, and length for next step
    #         input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    #         model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=config.is_encoder_decoder
    #         )
    #         unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

    #         # stop when each sentence is finished, or if we exceed the maximum length
    #         if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
    #             break
    #         yield input_ids
    
    # @torch.no_grad()
    # def stream_chat(model, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048,
    #                 do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    #     # if history is None:
    #     #     history = []
    #     if logits_processor is None:
    #         logits_processor = LogitsProcessorList()
    #     logits_processor.append(InvalidScoreLogitsProcessor())
    #     gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
    #                   "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    #     prompt = query
        
    #     inputs = tokenizer([prompt], return_tensors="pt").cuda()
    #     for outputs in stream_generate(**inputs, **gen_kwargs):
    #         outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    #         response = tokenizer.decode(outputs)
    #         response = process_response(response)
    #         yield response

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
            outputs = model.generate(
                        inputs,
                        # attention_mask=attention_mask,
                        max_length=max_length,
                        do_sample=True,
                        top_k = 5,
                        top_p=0.95,
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

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 2048, step=1
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
