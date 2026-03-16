from openai import OpenAI
from fastapi import FastAPI

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationTokenBufferMemory, ConversationBufferWindowMemory

llm_model = 'gpt-4o-mini-2024-07-18'

app = FastAPI(
    title="Hobby Buddy",
    description="A simple API that uses GPT to answer questions related to hobbies",
    version="1.0.0"
)

guitar_memory = ConversationBufferWindowMemory(k=2, memory_key="history")
guitar_template = """You are a very skilled guitarist and good guitar instructor. \
You are great at answering questions about learning and playing the guitar in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

History:
{history}

Here is a question:
{input}"""


photography_template = """You are a very good photographer. \
You are great at answering questions related to photography and how to get good shots. \
You are so good because you are able to undertand composition, color grading, 
and all the camera setups like shutter speed, aperture, etc. needed for a good photo. 
Give concise answers for a beginner.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name":"guitar",
        "description":"Good for answering questions about playing the guitar",
        "prompt_template": guitar_template
    },
    {
        "name":"photography",
        "description":"Good for answering questions about photography",
        "prompt_template": photography_template
    }
]

llm = ChatOpenAI(model=llm_model)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    if name == "guitar":
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["history","input"]
        )
        chain = LLMChain(llm=llm, prompt=prompt, memory=guitar_memory)
    else:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str = '\n'.join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template = router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)

router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

@app.get("/chat", summary="Ask a guitar or photography question")
def chat(prompt: str):
    """Route your question to the guitar or photography expert automatically."""
    response = chain.run(prompt)
    return {"response": response}

