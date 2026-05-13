import json
import os
import time
from openai import OpenAI
import base64

def gpt(messages):
    mess = []
    mess.append({"role": "user", "content": messages})
    client = OpenAI(api_key="", base_url="")

    response = client.chat.completions.create(
        model= "gpt-5-2025-08-07",
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

        # Take the first 15 pages of the paper content
        img_dir = os.path.join(files_dir, f"{item_id}")
        resized_imgs = []
        for file in os.listdir(img_dir):
            if file.lower().endswith(".png"):
                img_path = os.path.join(img_dir, file)
                resized_imgs.append(img_path)
        def extract_num_from_filename(path):
            filename = os.path.basename(path)
            num_str = os.path.splitext(filename)[0]
            return int(num_str)

        resized_imgs.sort(key=extract_num_from_filename)
        # print(len(resized_imgs))
        resized_imgs = resized_imgs[:15]

        part_idx = 0
        for qa in qa_pairs:
            part_idx += 1
            if qa["category"] != "Claim_Verification":
                prompt_template = "You are an expert academic assistant. Your task is to carefully read and analyze the provided complete research paper, and then answer the following question solely based on its content, arguments, and data, without using any external information or assumptions.\nResponse Requirements:\n1. The answer must be professional, precise, concise, and clearly presented.\n2. All statements in your answer must be exclusively derived from the paper's content and directly relevant to the question, avoiding any information or claims not supported by the paper.\n3. The total length of your response must not exceed 3000 characters (including spaces).\n\nQuestion:\n{question}"
            else:
                prompt_template = "You are an academic judgment specialist assigned to classify the following statement as strictly 'True' or 'False' based exclusively on the content of the provided research paper. Carefully read and analyze the entire paper. Use only evidence directly from the text—do not incorporate external knowledge, assumptions, or subjective reasoning.\n\nOutput Requirements:\n- Respond SOLELY with 'True' or 'False'\n- No explanations, disclaimers, or supplementary text\n\nStatement:\n{question}"
            messages = [{"type": "text", "text": prompt_template.format(question=qa["question"])}]

            for img_path in resized_imgs:
                # print(img_path)
                with open(img_path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode('utf-8')
                    messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}})
            
            post_data = gpt(messages)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'a', encoding='utf-8') as fw:
                fw.write(json.dumps({"id": item_id, "part_idx": part_idx, "question": qa["question"], "gen_answer": post_data, "category": qa["category"]}, ensure_ascii=False)+'\n')
            time.sleep(3)
    # fw.close()

if __name__ == "__main__":

    paper_qa("./benchmark/test.json", "./benchmark/vlm/test", "./output/gptV.json")
