import requests
import  json
from typing import List
url = "https://api.siliconflow.cn/v1/chat/completions"
api_key = "*****"

def api_qwen7b_generate( messages = None,   system_prompt = "",user_prompt =" ",temperature=1.0,top_k = 20,top_p=0.9,max_tokens = 512,type =  "text"):
    print("正在调用api")
    payload = {

        "model": "deepseek-ai/DeepSeek-V3",  #Pro/deepseek-ai/DeepSeek-V3
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": type}
    }

    if isinstance(messages,List) :
        payload["messages"] = messages
    else:
        payload["messages"] = [
            {
                "role": "system",
                "content": f"{system_prompt}"
            },
            {
                "role": "user",
                "content": f"{user_prompt}"
            }
        ],

    headers = {
        "Authorization": f"{api_key}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    response = json.loads(response.text)["choices"][0]["message"]["content"]
    print("api已返回")
    return response

if __name__=='__main__':
    prompt = [
    {
        "role": "system",
        "content": f"""谜底：飞机。用户的问题是：他有翅膀吗。
        请你根据谜底来回答人物的问题。你只能回答'是'或者'否'。
        人物会提出一个与谜底相关的问题。例如谜底是：苹果，人物问题是：“它是红色的吗？” 你需要回答'否'，因为苹果不一定是红色的。人物问题是：“它是否长在树上？” 你需要回答'是'，因为苹果是长在树上的。
        """
    }
    ]

    print(api_qwen7b_generate(messages=prompt))