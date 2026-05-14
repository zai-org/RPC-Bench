import json
import os
import re
import time

from openai import OpenAI

from prompt import (
    system_prompt_completeness_augment,
    system_prompt_conciseness_augment,
    system_prompt_correctness_augment,
    user_prompt_augment,
)


def gpt(messages):

    client = OpenAI(api_key="", base_url="")

    response = client.chat.completions.create(
        model="gpt-5-2025-08-07", messages=messages, stream=False
    )
    post_data = response.choices[0].message.content
    print("gpt: ", response.choices[0].message.content)

    return post_data


def gemini(messages):

    client = OpenAI(api_key="", base_url="")

    response = client.chat.completions.create(
        model="gemini-2.5-pro", messages=messages, stream=False
    )
    post_data = response.choices[0].message.content
    print("gemini: ", response.choices[0].message.content)

    return post_data


def escape_latex(json_str):
    json_str = re.sub(r'(?<!\\)\\(?![bfnrtu\\"\'\\])', r"\\\\", json_str)
    return json_str


def paper_qa_score(file_path, eval_path, out_path, judge_model="gpt"):

    gen_list = []
    for data in open(eval_path, "r", encoding="utf-8"):
        d = json.loads(data)
        gen_list.append(d)

    data_list = []
    paper_dict = {}
    for data in open(file_path, "r", encoding="utf-8"):
        d = json.loads(data)
        paper_dict[d["id"]] = {"title": d["title"], "abstract": d["abstract"]}
        for idx, qa in enumerate(d["qa_pairs"], start=1):
            qa["id"] = d.get("id")
            qa["part_idx"] = idx
            data_list.append(qa)
    print("gen: ", len(gen_list), "data: ", len(data_list))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for item, gen_item in zip(data_list, gen_list):
        assert item["id"] == gen_item["id"]
        assert item["part_idx"] == gen_item["part_idx"]
        assert item["question"] == gen_item["question"]

        title = paper_dict[item["id"]]["title"]
        abstract = paper_dict[item["id"]]["abstract"]

        score = []
        for sys_prompt in [
            system_prompt_conciseness_augment,
            system_prompt_completeness_augment,
            system_prompt_correctness_augment,
        ]:
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": user_prompt_augment.format(
                        title=title,
                        abstract=abstract,
                        question=item["question"],
                        reference_answer=item["answer"],
                        predicted_answer=gen_item["gen_answer"],
                    ),
                },
            ]
            if judge_model == "gpt":
                post_data = gpt(messages)
            else:
                post_data = gemini(messages)
            score.append(post_data)
            time.sleep(3)

        with open(out_path, "a", encoding="utf-8") as fw:
            fw.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "part_idx": item["part_idx"],
                        "question": item["question"],
                        "reference_answer": item["answer"],
                        "predicted_answer": gen_item["gen_answer"],
                        "category": item["category"],
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def LLM_judge(model):

    paper_qa_score(
        "./benchmark/test.json",
        f"./output/{model}.json",
        f"./output/score/gemini/{model}_score.json",
        judge_model="gemini",
    )
    paper_qa_score(
        "./benchmark/test.json",
        f"./output/{model}.json",
        f"./output/score/gpt/{model}_score.json",
        judge_model="gpt",
    )


def get_LLM_score(eval_path):

    category_dict = {}
    sum_c1, sum_c2, sum_c3, count = 0.0, 0.0, 0.0, 0
    for data in open(eval_path, "r", encoding="utf-8"):
        item = json.loads(data)
        cat = item["category"]
        if cat == "Claim_Verification":
            continue
        else:
            if cat not in category_dict:
                category_dict[cat] = {
                    "Conciseness": 0,
                    "Correctness": 0,
                    "Completeness": 0,
                    "count": 0,
                }

            content = {}
            for i_s in item["score"]:
                if i_s == "":
                    continue
                if "```json" in i_s:
                    pattern = r"```json(.*)```"
                    i_s = re.search(pattern, i_s, re.DOTALL).group(1)
                    i_s = escape_latex(i_s)
                i_s = json.loads(i_s, strict=False)
                content.update(i_s)

            category_dict[cat]["Conciseness"] += float(
                content.get("Conciseness", {}).get("rating", 0.00)
            )
            category_dict[cat]["Correctness"] += float(
                content.get("Correctness", {}).get("rating", 0.00)
            )
            category_dict[cat]["Completeness"] += float(
                content.get("Completeness", {}).get("rating", 0.00)
            )
            category_dict[cat]["count"] += 1

            sum_c1 += float(content.get("Conciseness", {}).get("rating", 0.00))
            sum_c2 += float(content.get("Correctness", {}).get("rating", 0.00))
            sum_c3 += float(content.get("Completeness", {}).get("rating", 0.00))
            count += 1

    result = {}
    for cat, values in category_dict.items():
        result[cat] = (
            values["Conciseness"] / values["count"],
            values["Correctness"] / values["count"],
            values["Completeness"] / values["count"],
        )

    print(count)
    print("Category | Conciseness | Correctness | Completeness | Informativeness")
    print(
        f"Overall: | {sum_c1/count*20:.3f} | {sum_c2/count*20:.3f} | {sum_c3/count*20:.3f} | {2*(sum_c2/count)*(sum_c3/count)/((sum_c2/count)+(sum_c3/count)+1e-8)*20:.3f}"
    )
    # print("---------------------------")
    for category in sorted(result.keys()):
        Conciseness, Correctness, Completeness = result[category]
        if Correctness + Completeness == 0:
            Informativeness = 0
        else:
            Informativeness = (
                2 * Correctness * Completeness / (Correctness + Completeness)
            )
        print(
            f"{category:4} | {Conciseness*20:.3f}  | {Correctness*20:.3f}  | {Completeness*20:.3f}  | {Informativeness*20:.3f}"
        )

    total_scores = {
        "Conciseness": sum_c1 / count,
        "Correctness": sum_c2 / count,
        "Completeness": sum_c3 / count,
    }
    return total_scores, result


def evaluate_two_files(model):
    print(
        "-----------------------------------------GPT Judge-----------------------------------------"
    )
    total1, result1 = get_LLM_score(f"./output/score/gpt/{model}_score.json")
    print(
        "-----------------------------------------Gemini Judge-----------------------------------------"
    )
    total2, result2 = get_LLM_score(f"./output/score/gemini/{model}_score.json")

    avg_total = {k: (total1[k] + total2[k]) / 2 for k in total1.keys()}

    avg_result = {}
    all_cats = set(result1.keys()) | set(result2.keys())
    for cat in all_cats:
        if cat in result1 and cat in result2:
            avg_c = (result1[cat][0] + result2[cat][0]) / 2
            avg_cor = (result1[cat][1] + result2[cat][1]) / 2
            avg_com = (result1[cat][2] + result2[cat][2]) / 2
        elif cat in result1:
            avg_c, avg_cor, avg_com = result1[cat]
        else:
            avg_c, avg_cor, avg_com = result2[cat]
        avg_result[cat] = (avg_c, avg_cor, avg_com)

    print(
        f"-----------------------------------------{model} Final Score-----------------------------------------"
    )
    print(
        "Category | Conciseness | Correctness | Completeness | F1-like | Informativeness"
    )
    info = (
        2
        * avg_total["Correctness"]
        * avg_total["Completeness"]
        / (avg_total["Correctness"] + avg_total["Completeness"] + 1e-8)
    )
    print(
        f"Overall: | {avg_total['Conciseness'] * 20:.2f} | {avg_total['Correctness'] * 20:.2f} | {avg_total['Completeness'] * 20:.2f} | {info * 20:.2f} | {avg_total['Conciseness'] * info * 4:.2f}"
    )
    for cat in sorted(avg_result.keys()):
        c1, c2, c3 = avg_result[cat]
        info = 0 if (c2 + c3) == 0 else (2 * c2 * c3 / (c2 + c3))
        print(
            f"{cat:4} | {c1*20:.2f}  | {c2*20:.2f}  | {c3*20:.2f}  | {info*20:.2f}  | {c1*info*4:.2f}"
        )


def calculate_acc(pred, ground_true):

    assert len(pred) == len(ground_true), "List lengths do not match"
    assert all(
        p in ("True", "False") for p in pred
    ), "Prediction list contains invalid values"
    assert all(
        gt in ("True", "False") for gt in ground_true
    ), "Ground truth list contains invalid values"

    total_correct = sum(1 for p, gt in zip(pred, ground_true) if p == gt)
    overall_acc = total_correct / len(pred) if pred else 0.0

    return overall_acc


def get_verification_score(gold_path, eval_path):

    gold_answers = []
    eval_answers = []
    gold_fact = []
    eval_fact = []
    for data in open(gold_path, "r", encoding="utf-8"):
        d = json.loads(data)
        qa_pairs = d["qa_pairs"]
        for qa in qa_pairs:
            if qa["category"] != "Claim_Verification":
                continue
            gold_answers.append(qa["answer"])

    for data in open(eval_path, "r", encoding="utf-8"):
        d = json.loads(data)
        if d["category"] != "Claim_Verification":
            continue
        eval_answers.append(d["gen_answer"])

    assert len(gold_answers) == len(eval_answers)
    print("Claim_Verification nums: ", len(gold_answers))

    for g, e in zip(gold_answers, eval_answers):
        if e not in ["True", "False"]:
            continue

        gold_fact.append(g)
        # eval_fact.append(e)
        if e not in ["True", "False"]:
            if g == "True":
                eval_fact.append("False")
            else:
                eval_fact.append("True")
        else:
            eval_fact.append(e)

    results = calculate_acc(eval_fact, gold_fact)
    print("---------------------------")
    print("Claim_Verification")
    print(f"ACC:        {results:.2%}")


# Example usage
if __name__ == "__main__":

    # LLM_judge("gpt")
    evaluate_two_files("gpt")
    # get_verification_score("./benchmark/test.json", "./output/gpt.json")
