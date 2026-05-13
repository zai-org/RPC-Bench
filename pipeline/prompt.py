decompose_prompt = '''You are an excellent reviewer of papers. You are tasked with extracting QA pairs from the "review", "rebuttal" and "extra_rebuttal" sections of a conference paper submission. This process includes identifying "review" provided by reviewers and pairing them with the corresponding answers authored by the paper's authors, utilizing content from both the "rebuttal" and any relevant "extra_rebuttal" sections. 
    Your goals are: Extract and classify the QA pairs. Ensure that references and citations in the rebuttal are preserved in their original format within the answers, maintaining the academic rigor and clarity. Determine whether each question-answer pair is 'multimodal-related,' a broad concept that includes questions explicitly about the figures and tables in the paper or questions that can only be answered by referring to the contents of these figures and tables.

    Input Structure:

    review: Concatenation of all reviews, including multifaceted evaluations of the paper and any responses or questions directed at the authors' rebuttal.
    rebuttal: The content in the rebuttal is a concatenation of the answers to all the review questions.
    extra_rebuttal: Additional content from the authors that may cover the current questions.

    Output Requirements:

    For each QA pair, output in the following JSON format:
    [
        {
            "question": "extracted question text here",
            "answer": "corresponding answer text here",
            "is_multimodal_related": true or false
        },
        ...
    ]

    Guidelines:

    1. Split combined questions into finer sub-questions for clarity but merge them if they cannot stand alone meaningfully.
    2. Ensure the completeness and consistency of the extracted QA pairs.
    3. Use content from the extra_rebuttal to enhance or clarify answers when applicable and relevant to the question.
    4. Ensure that the rebuttal content is fully utilized in the answers, forming comprehensive and clear QA pairs that correspond to the questions posed.
    5. Use your judgment to label each QA pair as 'multimodal-related' if it either explicitly poses questions about the figures and tables in the paper or implicitly requires the content of these figures and tables to answer the question.
    6. The answers should be as comprehensive as possible, retaining any relevant content such as "references" that can assist in addressing the questions.
    7. Use the original content from the review, rebuttal, and extra_rebuttal to construct the QA pairs, avoiding unnecessary modifications to the original text.

    Input:
    review: It is novel enough to combine the advantages of two famous models (Transformer, RNN). Also, the combining method looks applicable to a variety of scenarios. The experimental results are impressive, showing superior performance to previous Transformer.

    I think the draft would become better if there is a more complete explanation and figures about the self-attention with recurrence (RSA) operation.

    I think the novelty of this draft is enough for the publication and the experimental results are impressive. English is good enough as well. I recommend weak accept for the draft.


    rebuttal: Thanks for your encouraging words and constructive comments. We sincerely appreciate your time in reading the paper, and our point-to-point responses to your comments are given below.

    > I think the draft would become better if there is a more complete explanation and figures about the self-attention with recurrence (RSA) operation.

    Thank you for this instructive comment. Following your suggestions, we have provided a graphical illustration of a single headed RSA module in Figure 1 (d) on Page 2, and a more detailed explanation about the operation of RSA has been given in the paragraph of "Operation of multihead RSA modules" on Page 5.       

    In the meanwhile, we have also reorganized the whole Section 3 to better explain the proposed RSA. Specifically,
    For a single head RSA, we have devoted a paragraph right after equation (4) to detail the different types of REMs i.e. $\mathbf{P}$ in the paper.

    For your easy reference, we have listed the multihead RSA operation below:

            Procedure for the Multihead RSA
                    - Choose masked or unmasked REMs according to the nature of the task.
                    - Select the hyperparameters including the dilating factor $d$ and the numbers of the six types of REMs $(k_1,\dots,k_6)$.
                    - For each head, apply equation (4) with a different REM.
                    - Apply a linear layer to combine the output from all heads, and perform layer-normalization and dropout.

    extra_rebuttal: We will make the following revisions to the paper:

    1. Block-Recurrent Transformer (BRT) [1] has been adopted as another baseline model for the NLP experiment in Section 4.3, and its results are presented as follows.

    |                             | BRT        | RSA-BRT    |
    | --------------------------- | ---------- | ---------- |
    | Enwik8                      | 1.0746     | **1.0683** |
    | Text8                       | 1.1652 | **1.1625** |
    | WikiText-103                | 23.758     | **23.639** |
    | # Averaged Params added (%) |            | 8.68E-05   |

    It can be seen that RSA-BRT exceeds the baseline BRT's performance on all datasets.

    **The results of this table will be used to fill in the blanks in Table 3 (b) of the paper.**



    2. Two additional experiments for Section 4.4 have been conducted during the second discussion phase, which are detailed in the responses to Reviewers mvWh and Zrmk.

    (1) A scaling experiment is conducted for RSA-BRT v/s BRT on Enwik8 dataset. The results are shown as follows.

    | #  layers           | 8          |           | 10         |           | 12         |           | 14         |           |
    | ------------------- | ---------- | --------- | ---------- | --------- | ---------- | --------- | ---------- | --------- |
    |                     | Params     | BPC       | Params     | BPC       | Params     | BPC       | Params     | BPC       |
    | BRT                 | 35,080,908 | 1.127     | 41,905,868 | 1.106     | 48,730,828 | 1.098     | 55,555,788 | 1.079     |
    | RSA-BRT             | 35,080,943 | **1.120** | 41,905,913 | **1.104** | 48,730,883 | **1.092** | 55,555,853 | **1.072** |
    | Increase in #Params | 35         |           | 45         |           | 55         |           | 65         |           |

    It can be seen that, with only less than 100 new parameters, RSA-BRT can achieve some improvement over the baseline BRT. More importantly, the advantage can be consistently observed for all model sizes.     



    (2) Another scaling experiment is conducted for RSA-XL against TL-XL on Text8 dataset, where REM is replaced by a learnable Toeplitz matrix in the latter model. The results are shown as follows.

    | #  layers           | 8          |           | 10         |           | 12         |           | 14         |           |
    | ------------------- | ---------- | --------- | ---------- | --------- | ---------- | --------- | ---------- | --------- |
    |                     | Params     | BPC       | Params     | BPC       | Params     | BPC       | Params     | BPC       |
    | TL-XL               | 34,180,645 | 1.193     | 41,013,799 | 1.188     | 47,846,953 | 1.183     | 54,680,107 | 1.178     |
    | RSA-XL              | 34,139,725 | **1.181** | 40,964,695 | **1.170** | 47,789,665 | **1.164** | 54,614,635 | **1.160** |
    | Decrease in #Params | 40,920     |           | 49,104     |           | 57,288     |           | 65,472     |           |

    From the above table, it can be seen that the newly added TL-XL also performs worse than the RSA-XL of a similar model size, indicating parameter redundancy. In other words, RSA-XL enjoys a much better parameter-efficiency.

    **These two experiments will be further included into Section 4.4 of the paper.**



    Reference

    [1] Hutchins, D., Schlag, I., Wu, Y., Dyer, E., and Neyshabur, B. (2022). Block-recurrent transformers. In Advances in Neural Information Processing Systems.

    Output:
    [
        {
            "question": "I think the draft would become better if there is a more complete explanation and figures about the self-attention with recurrence (RSA) operation.",
            "answer": "Thank you for this instructive comment. Following your suggestions, we have provided a graphical illustration of a single headed RSA module in Figure 1 (d) on Page 2, and a more detailed explanation about the operation of RSA has been given in the paragraph of 'Operation of multihead RSA modules' on Page 5. In the meanwhile, we have also reorganized the whole Section 3 to better explain the proposed RSA. Specifically, for a single head RSA, we have devoted a paragraph right after equation (4) to detail the different types of REMs i.e. $\\mathbf{P}$ in the paper. For your easy reference, we have listed the multihead RSA operation below: Procedure for the Multihead RSA - Choose masked or unmasked REMs according to the nature of the task. - Select the hyperparameters including the dilating factor $d$ and the numbers of the six types of REMs $(k_1,\\dots,k_6)$. - For each head, apply equation (4) with a different REM. - Apply a linear layer to combine the output from all heads, and perform layer-normalization and dropout.",
            "is_multimodal_related": true
        }
    ]

    Input:
    review: I would like to request further clarification regarding your paper after carefully reading it. Firstly, I would like to express my sincere appreciation for the captivating nature of your work and the clarity with which it is presented. Congratulations for the acceptance of your paper into the top 5% category.

    In Section 4.3, I noticed the utilization of Transformer-XL with 14 layers, resulting in a notable achievement of 1.074 on the Enwik8 dataset. However, upon referencing the Transformer-XL paper, it became apparent that they reported lower bpc values, specifically 1.06 with 12 layers, 1.03 bpc with 18 layers, and an impressive 0.99 bpc with 24 layers.

    To enhance my understanding, I kindly request your insights regarding the decision to opt for 14 layers instead and the possible reasons behind the relatively higher bpc despite employing deeper layers. Additionally, I would greatly appreciate any additional details or insights you can provide to address these inquiries.

    Thank you in advance for your time and consideration. Your input will greatly contribute to my comprehension of your valuable research. Once again, congratulations on the successful publication of your paper.
    rebuttal: Hi Lokesh, thanks for the question!

    The observed difference between the reported bits per character (bpc) for Enwik8 in Section 4.3 of our paper and the original Transformer-XL paper can be attributed to our decision to utilize Nvidia's implemented Transformer-XL (https://catalog.ngc.nvidia.com/orgs/nvidia/resources/transformerxl_for_pytorch) rather than the official repository. We chose the Nvidia version due to its enhanced user-friendliness and comprehensive multi-card support.

    However, it is important to note that the reproduction by Nvidia resulted in slightly worse bpc for Enwik8 compared to the figures reported in the original paper. Specifically, the bpc for Enwik8 with a 12-layer Transformer-XL exceeded the previously reported value of 1.06. This discrepancy could be attributed to variations in the implementation and environment between Nvidia's version and the official repository.

    Furthermore, from an intuitive perspective, when a model is overparameterized, the proposed RSA may exhibit better generalization ability, as illustrated in Figure 1. In order to emphasize the benefits of the proposed RSA, we employed a slightly larger model. Unfortunately, due to limited resources, we were unable to conduct further experiments using a 24-layer XL model.

    While acknowledging these limitations, we believe that the use of Nvidia's implementation, combined with our modifications, provides valuable insights and supports our argument. The comparison between the modified models, despite the slight deviations, offers meaningful observations regarding the potential advantages of the proposed RSA.

    extra_rebuttal:
    Output:
    [
        {
            "question": "In Section 4.3, I noticed the utilization of Transformer-XL with 14 layers, resulting in a notable achievement of 1.074 on the Enwik8 dataset. However, upon referencing the Transformer-XL paper, it became apparent that they reported lower bpc values, specifically 1.06 with 12 layers, 1.03 bpc with 18 layers, and an impressive 0.99 bpc with 24 layers. To enhance my understanding, I kindly request your insights regarding the decision to opt for 14 layers instead and the possible reasons behind the relatively higher bpc despite employing deeper layers.",
            "answer": "The observed difference between the reported bits per character (bpc) for Enwik8 in Section 4.3 of our paper and the original Transformer-XL paper can be attributed to our decision to utilize Nvidia's implemented Transformer-XL (https://catalog.ngc.nvidia.com/orgs/nvidia/resources/transformerxl_for_pytorch) rather than the official repository. We chose the Nvidia version due to its enhanced user-friendliness and comprehensive multi-card support. However, it is important to note that the reproduction by Nvidia resulted in slightly worse bpc for Enwik8 compared to the figures reported in the original paper. Specifically, the bpc for Enwik8 with a 12-layer Transformer-XL exceeded the previously reported value of 1.06. This discrepancy could be attributed to variations in the implementation and environment between Nvidia's version and the official repository. Furthermore, from an intuitive perspective, when a model is overparameterized, the proposed RSA may exhibit better generalization ability, as illustrated in Figure 1. In order to emphasize the benefits of the proposed RSA, we employed a slightly larger model. Unfortunately, due to limited resources, we were unable to conduct further experiments using a 24-layer XL model. While acknowledging these limitations, we believe that the use of Nvidia's implementation, combined with our modifications, provides valuable insights and supports our argument. The comparison between the modified models, despite the slight deviations, offers meaningful observations regarding the potential advantages of the proposed RSA.",
            "is_multimodal_related": true
        }
    ]
    
    Input:
    review: -       The idea of utilizing dataset exchangeability to identify test set contamination is novel and interesting.
    -       The proposed sharded likelihood comparison test addresses the tradeoff between statistical power and computational requirements of the permutation test, which is promising. The sharded rank comparison test also provides (asymptotic) guarantees on false positive rates.
    -       Experimental results are promising. A GPT-2 model is trained from scratch on standard pretraining data and known test sets to verify the efficiency of the proposed method in identifying test set contamination. The method is also tested with an existing model, LLaMA2, on the MMLU dataset, showing general agreement with the contamination study results.   
    -       Although a more efficient sharded rank comparison test is proposed, the computational complexity is still considerable. For example, testing 49 files using 1000 permutations per shard can take 12 hours for LLaMA2.
    -       There is no comparison with other baseline methods.
    -       The method relies on a strong assumption of data exchangeability, which may not hold in real-world datasets.     
    If a dataset is not exchangeable, how effective is the method?

    rebuttal: Thank you for your thorough review and valuable feedback on our work.

    We'd like to address the concern regarding the computational complexity of our test. It's important to note that the test is a one-time process for any given model and dataset; once the p-values are computed, there is no need for recalculation. Our findings indicate that a number of permutations beyond 30-50 per shard offers diminishing returns, as shown in Figure 3 (right).

    Furthermore, the test's design allows for easy parallelization. Each shard permutation can be evaluated independently, enabling the use of inexpensive commodity hardware to run the test significantly faster.

    Regarding the assumption of data exchangeability, this is a strictly weaker condition than the commonly held assumption of independent and identically distributed (I.I.D.) data in machine learning. Most datasets satisfy this assumption to some extent.

    We acknowledge the validity of our test hinges on data exchangeability. However, depending on the source of non-exchangeability, it is often the case that a dataset can be altered slightly so that our test is still valid. For example, a common source of non-exchangeability is the presence of ascending IDs (e.g. as in SQuAD and HumanEval). We can adjust the data—by either removing these IDs or permuting the examples while keeping IDs constant—to retain the test's applicability. This is discussed in more detail in the revised paper.

    Finally, we appreciate your suggestion to include baseline comparisons. We provide a comparison against a contamination detection method called Min-K% Prob, a state of the art heuristic method for contamination detection in language models proposed contemporaneous to our work by Shi et. al. (2023).

    We find that our method matches or exceeds the performance of this state of the art heuristic method. Please see the table in the top-level comment for numbers.

    extra_rebuttal: We are sincerely grateful to the reviewers for dedicating their time and effort to review our work, and we appreciate the recognition of the novelty of using exchangeability for contamination detection and the significance of our contribution given the discourse surrounding contamination in the field. We address each reviewer's comments in detail below. We have made numerous updates to the submission, most notably with the results of our test on four popular open models and eight commonly used benchmarks.

    One question shared by multiple reviewers is regarding the exact notion of contamination we consider in this work. Rather than consider a definition based on heuristics like n-gram overlap, we consider contamination detection as the problem of detecting statistical dependence between the test data and model parameters. Within this setting, our work shows that it is possible to provide provable guarantees of contamination in the case of verbatim contamination, where the full test set (with examples and labels) is embedded in the pretraining data.

    To illustrate the relevance of this setting, we note that a search of The Pile, a large open-source language modeling dataset, yielded numerous instances of small real-world datasets embedded with examples appearing in-order. As one example, the following is an excerpt from a dataset for an annotation tool made by Explosion, the creators of spaCy, a popular natural language processing framework, found in The Pile:

    ```
    {"text":"Uber\u2019s Lesson: Silicon Valley\u2019s Start-Up Machine Needs Fixing","meta":{"source":"The New York Times"}}
    {"text":"Pearl Automation, Founded by Apple Veterans, Shuts Down","meta":{"source":"The New York Times"}}
    {"text":"How Silicon Valley Pushed Coding Into American Classrooms","meta":{"source":"The New York Times"}}

    Source: https://github.com/explosion/prodigy-recipes/tree/fc06f6a6d93bc477e98cf0d8357c39322e4f5a6a
    ```

    What our work shows is that by exploiting exchangeability in this setting, we are able to provide guarantees on the false positive rate of our test.

    Multiple reviewers indicated the desire for a comparison against a baseline method. While no other existing work is comparable in the sense that it provides a statistical proof of contamination like ours, we provide a comparison against a state of the art heuristic method for contamination detection called Min-K% Prob, proposed by Shi et. al. (2023) contemporaneous to our work. We use the same pretrained model and test sets from our experiments in Section 4.1.

    | Dataset    | Duplication Count | Sharded p (ours) | Percent Contaminated (Min-K%-Prob) |
    |------------|-------------------|------------------|------------------------------------|
    | BoolQ     | 1            | 0.156          | 3%                             |
    | HellaSwag | 1            | 0.478          | 2%                            |
    | MNLI     | 10            | 1.96e-11           | 100%                           |
    | MMLU-Pro-Law | 50        |  1e-38           | 90%                         |
    | MMLU-HS-Psych | 100     |  1e-38           | 74% |

    Our run of Min-k%-Prob follows the methodology outlined in the paper; we run the method on one hundred 512-token spans sampled from each benchmark, and tune the decision threshold on a validation set of five of our contaminated test sets, and five test sets not used in our data mixture (uncontaminated). The threshold is tuned for a false positive rate of 5% to allow for a meaningful comparison against our test. A value of k=20 is used as is recommended in the paper.

    We find that our method matches or exceeds the performance of this state of the art heuristic method, while also providing statistical proof of contamination.

    Output:
    [
        {
            "question": "Although a more efficient sharded rank comparison test is proposed, the computational complexity is still considerable. For example, testing 49 files using 1000 permutations per shard can take 12 hours for LLaMA2.",
            "answer": "We'd like to address the concern regarding the computational complexity of our test. It's important to note that the test is a one-time process for any given model and dataset; once the p-values are computed, there is no need for recalculation. Our findings indicate that a number of permutations beyond 30-50 per shard offers diminishing returns, as shown in Figure 3 (right). Furthermore, the test's design allows for easy parallelization. Each shard permutation can be evaluated independently, enabling the use of inexpensive commodity hardware to run the test significantly faster.",
            "is_multimodal_related": true
        },
        {
            "question": "There is no comparison with other baseline methods.",
            "answer": "Finally, we appreciate your suggestion to include baseline comparisons. We provide a comparison against a contamination detection method called Min-K% Prob, a state of the art heuristic method for contamination detection in language models proposed contemporaneous to our work by Shi et. al. (2023). We find that our method matches or exceeds the performance of this state of the art heuristic method. Please see the table in the top-level comment for numbers. While no other existing work is comparable in the sense that it provides a statistical proof of contamination like ours, we provide a comparison against a state of the art heuristic method for contamination detection called Min-K% Prob, proposed by Shi et. al. (2023) contemporaneous to our work. We use the same pretrained model and test sets from our experiments in Section 4.1.\n\n| Dataset    | Duplication Count | Sharded p (ours) | Percent Contaminated (Min-K%-Prob) |\n|------------|-------------------|------------------|------------------------------------|\n| BoolQ     | 1            | 0.156          | 3%                             |\n| HellaSwag | 1            | 0.478          | 2%                            |\n| MNLI     | 10            | 1.96e-11           | 100%                           |\n| MMLU-Pro-Law | 50        |  1e-38           | 90%                         |\n| MMLU-HS-Psych | 100     |  1e-38           | 74% |\n\nOur run of Min-k%-Prob follows the methodology outlined in the paper; we run the method on one hundred 512-token spans sampled from each benchmark, and tune the decision threshold on a validation set of five of our contaminated test sets, and five test sets not used in our data mixture (uncontaminated). The threshold is tuned for a false positive rate of 5% to allow for a meaningful comparison against our test. A value of k=20 is used as is recommended in the paper. We find that our method matches or exceeds the performance of this state of the art heuristic method, while also providing statistical proof of contamination.",
            "is_multimodal_related": false
        },
        {
            "question": "The method relies on a strong assumption of data exchangeability, which may not hold in real-world datasets.",
            "answer": "Regarding the assumption of data exchangeability, this is a strictly weaker condition than the commonly held assumption of independent and identically distributed (I.I.D.) data in machine learning. Most datasets satisfy this assumption to some extent.",
            "is_multimodal_related": false
        },
        {
            "question": "If a dataset is not exchangeable, how effective is the method?",
            "answer": "We acknowledge the validity of our test hinges on data exchangeability. However, depending on the source of non-exchangeability, it is often the case that a dataset can be altered slightly so that our test is still valid. For example, a common source of non-exchangeability is the presence of ascending IDs (e.g. as in SQuAD and HumanEval). We can adjust the data—by either removing these IDs or permuting the examples while keeping IDs constant—to retain the test's applicability. This is discussed in more detail in the revised paper.",
            "is_multimodal_related": false
        }
    ]
    '''


rewrite_prompt = '''You are an advanced assistant trained for academic research purposes. Your task is to process all review-rebuttal pairs into a structured Question-Answer (QA) format. For every input pair, follow these instructions:

    Input Structure:
    You will process all review-rebuttal pairs, where each is provided in the following format:
    Review: A statement or query from a reviewer providing feedback or posing a question about the submission.
    Rebuttal: The corresponding author response addressing the feedback.

    Processing Instructions:
    For each review-rebuttal pair, follow the steps below in strict sequence:
    1. Extract the Question (Q):
    Reformulate the reviewer feedback into a clear, precise, and standalone question. Ensure the question:
    Includes all necessary context from both the review and rebuttal (e.g., clarify vague references such as "this figure" or "the results").
    Is phrased in neutral and objective language, avoiding subjective or opinionated terms.
    2. Extract the Answer (A):
    Reformulate the author's rebuttal into a concise, objective, and standalone answer. Ensure the answer:
    Directly addresses the reformulated question.
    Is based strictly on the rebuttal content. Avoid additional interpretations, subjective language, or opinions.
    3. Classify the Question:
    Assign exactly one category from the following candidate categories. The value of "Category" must exactly match one of these labels:
    - Concept Understanding
    - Method Disambiguation
    - Method Mechanics
    - Motivation Analysis
    - Method Comparison
    - Experimental Exposition
    - Experimental Setup
    - Experimental Analysis
    - Claim Verification

    Category Definitions:
    1. Concept Understanding:
    Questions that clarify concepts, terminology, theoretical viewpoints, assumptions, or information
    conveyed in figures, tables, or formulas.

    2. Method Disambiguation:
    Questions that resolve ambiguity or misunderstanding about the proposed method, its scope, or its
    distinction from related concepts.

    3. Method Mechanics:
    Questions about how a method, model component, algorithmic step, module, loss, training procedure,
    or implementation detail works.

    4. Motivation Analysis:
    Questions about why a method, design choice, assumption, experiment, dataset, or modeling decision
    was used.

    5. Method Comparison:
    Questions comparing the proposed method with baselines, prior work, alternative designs, or
    competing approaches.

    6. Experimental Exposition:
    Questions asking what experimental results, tables, figures, metrics, or observations show,
    including direct explanation of reported results.

    7. Experimental Setup:
    Questions about experimental design, datasets, evaluation protocol, hyperparameters, data splits,
    baselines, metrics, or implementation settings.

    8. Experimental Analysis:
    Questions asking why certain experimental results occurred, how to interpret them, whether they
    generalize, or what conclusions can be drawn from them.

    9. Claim Verification:
    Binary factual verification questions that judge whether a claim, assumption, experimental
    conclusion, implementation detail, or completeness statement is correct based on the rebuttal.

    Special Rules for Claim Verification:
    - If "Category" is "Claim Verification", the generated question must be a clear, self-contained
    True/False question.
    - The answer "A" must be exactly either "True" or "False".
    - Do not use explanatory sentences in "A" for Claim Verification.
    - The question should focus on objective scientific, technical, or experimental facts.
    - Avoid subjective or evaluative phrasing such as "Did the authors adequately address..." or "Was
    the response satisfactory..."
    - Prefer factual formulations such as:
        - "Was RAFT tested on text-to-SQL or reasoning-based QA tasks?"
        - "Is demographic information beyond aggregated country-level selections available?"
        - "Does the rebuttal state that the same training data budget was used for all compared
    methods?"
    - Determine True or False strictly from the rebuttal content.

    Output Format:
    Return only valid JSON in the following format:
    [
        {
            "review": "Original reviewer feedback",
            "rebuttal": "Original author rebuttal",
            "Q": "Generated standalone question",
            "A": "Generated answer, or exactly True/False if Category is Claim Verification",
            "Category": "One of the allowed category labels"
        }
    ]

    Important Constraints:
    - Output JSON only. Do not include explanations outside the JSON.
    - The "Category" field must exactly match one of the nine allowed labels.
    - If the selected category is not "Claim Verification", the answer should be a concise natural-
    language answer.
    - If the selected category is "Claim Verification", the answer must be exactly "True" or "False".
    - Do not invent facts beyond the rebuttal.
    '''


quality_filter_prompt = '''You are a strict quality filter for OpenReview-derived academic data.

Your task is to decide whether an item should be kept for building an academic QA benchmark.
The item may come from either:
1. the comment-response decomposition stage, where the input is a reviewer comment and an author response; or
2. the QA rewriting stage, where the input is a generated QA item with its original review-rebuttal evidence.

The item must be reliably answerable using only the paper content and the provided review/rebuttal or QA evidence. Remove low-quality items that do not contain academic substance.

Filtering Criteria:
Remove the item if it falls into any of the following categories:

1. Temporary or editorial issue:
- Grammar, spelling, typo, wording, formatting, color/font/figure style changes.
- Adding a citation/reference or open-sourcing code/data when the response or answer merely acknowledges the change.
- Examples: "We corrected 'benchamrks' to 'benchmarks'"; "We added the reference to Smith et al."; "We will release code."

2. External resource dependency:
- The response or answer depends on external URLs, external papers, supplemental material not included in the paper, or tells the reader to look elsewhere without a substantive answer.
- Indirect or evasive replies such as "See Section X" without giving the relevant content.
- Examples: "More cases: https://..."; "Please refer to our website"; "See Appendix D" with no concrete details.

3. Non-substantive commitments:
- Promises future additions or revisions without providing specific details or a concrete resolution in the current submission.
- Examples: "We will add a limitations section"; "We will address this in future work"; "We will clarify this later."

Keep the item only if it provides substantive academic content, such as methodological details, conceptual clarification, experimental evidence, quantitative results, concrete analysis, or a verifiable claim grounded in the provided evidence.

Return only valid JSON:
{
  "keep": true or false,
  "reason": "short reason for the decision",
  "filter_type": "none | temporary_or_editorial_issue | external_resource_dependency | non_substantive_commitment"
}
'''
