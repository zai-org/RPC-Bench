import json
import os
import time
from openai import OpenAI

def gpt(messages):
    mess = []
    mess.append({"role": "user", "content": messages})
    client = OpenAI(api_key="", base_url="")

    response = client.chat.completions.create(
        model="gpt-5-2025-08-07",
        messages=mess,
        stream=False
    )
    post_data = response.choices[0].message.content
    print("gpt: ", response.choices[0].message.content)

    return post_data

def paper_qa(file_path, files_dir, output_path):

    for data in open(file_path, 'r', encoding='utf-8'):
        d = json.loads(data)
        item_id = d["id"]
        qa_pairs = d["qa_pairs"]

        content = None
        file_path = os.path.join(files_dir, item_id, f"{item_id}.md")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        else:
            raise FileNotFoundError(f"No .md file corresponding to id '{item_id}' was found in directory '{files_dir}'")

        part_idx = 0
        for qa in qa_pairs:
            part_idx += 1
            if qa["category"] != "Claim_Verification":
                prompt_template = "You are an expert academic assistant. Your task is to carefully read and analyze the provided complete research paper, and then answer the following question solely based on its content, arguments, and data, without using any external information or assumptions.\nResponse Requirements:\n1. The answer must be professional, precise, concise, and clearly presented.\n2. All statements in your answer must be exclusively derived from the paper's content and directly relevant to the question, avoiding any information or claims not supported by the paper.\n3. The total length of your response must not exceed 3000 characters (including spaces).\n\nQuestion:\n{question}\n\nPaper:\n{content}"
            else:
                prompt_template = "You are an academic judgment specialist assigned to classify the following statement as strictly 'True' or 'False' based exclusively on the content of the provided research paper. Carefully read and analyze the entire paper. Use only evidence directly from the text—do not incorporate external knowledge, assumptions, or subjective reasoning.\n\nOutput Requirements:\n- Respond SOLELY with 'True' or 'False'\n- No explanations, disclaimers, or supplementary text\n\nStatement:\n{question}\n\nPaper:\n{content}"
            messages = prompt_template.format(question=qa["question"], content=content)
            post_data = gpt(messages)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'a', encoding='utf-8') as fw:
                fw.write(json.dumps({"id": item_id, "part_idx": part_idx, "question": qa["question"], "gen_answer": post_data, "category": qa["category"]}, ensure_ascii=False)+'\n')
            time.sleep(5)

if __name__ == "__main__":
    
    paper_qa("./benchmark/test.json", "./benchmark/md/test", "./output/gpt.json")
    
