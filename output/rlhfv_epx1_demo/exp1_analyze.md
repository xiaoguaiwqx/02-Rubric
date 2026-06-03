## exp1 init critiq-rlhf-v

将critiq 适配多模态输入。**worker 有多模态输入，但是 manager 还是纯文本情况**。

**模型：** VLLM server Qwen/Qwen3-VL-8B-Instruct

**数据集**：90 条 discovery 数据里：

- train_set: 前 50 条，用于优化 criteria；

- valid_set: 后 40 条，只用于观察泛化，不参与当前轮 criteria 改写。

**进入主优化循环。每一轮大致做：**

1. 在 train_set 上评估每条 criterion；

2. score >= 0.8 的认为 good，保留；

3. 0.6 < score < 0.8 的认为 mid，让 manager 看错例后改写；

4. score <= 0.6 的认为 low，移除并生成新 criterion；
5. 可选地在 valid_set 上观察一轮。

**raw data:**

| 阶段   | Train      | Valid       |
| ------ | ---------- | ----------- |
| warmup | -          | 0.65 26/40  |
| iter 0 | 0.54 27/50 | 0.60 24/40  |
| iter 1 | 0.50 25/50 | 0.70 28/40  |
| iter 2 | 0.56 28/50 | 0.65 26/40  |
| iter 3 | 0.56 28/50 | 0.625 25/40 |
| iter 4 | 0.50 25/50 | 0.65 26/40  |
| final  | 0.52 26/50 | 0.65 26/40  |

**注意**：对于每一个iter Valid 都是使用 current criterion 来验证，在final的时候**使用的是 score >= 0.6 的criterion来作验证**，在这次实验 score >= 0.6 的criterion 刚好有5个。

**初始化的五个criterion**, 初始化方法：

- 这不是让 manager 直接输出 criteria，而是先给它少量有标签的 A/B 样本预热先。manager 会先解释这些偏好原因。作为历史上下文。
- 解释完成后，它再根据这些“刚看过的例子”，生成更贴近 RLHF-V 的 criteria。

| criterion                                                    | score | description                                                  |
| ------------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| visual_grounding<br/>视觉定位                                | 0.568 | The answer must be directly supported by visible elements in the image. Annotators prefer answers that explicitly reference observable objects, positions, colors, or actions — and penalize those that make unsupported inferences or invent details not present in the image. |
| question_relevance <br/>问题相关性                           | 0.532 | The answer must directly address the question without straying into irrelevant topics or adding extraneous information. Annotators prefer responses that focus on what is asked, even if the answer is negative or ambiguous — as long as it’s directly responsive. |
| factual_consistency_with_the_image<br/>与图像的事实一致性    | 0.574 | The answer must not contradict or misrepresent what is visibly present. Annotators penalize answers that misidentify objects, people, or actions — for example, claiming someone is taking a photo when the camera is idle, or mislabeling a bird species without visual evidence. |
| avoidance_of_hallucinated_visual_details<br/>避免虚构视觉细节 | 0.442 | The answer must not invent or fabricate visual elements that are not present in the image. Annotators prefer cautious, evidence-based descriptions — even if they are vague — over confident claims that hallucinate details (e.g., ‘a red car is parked behind the building’ when none is visible). |
| calibrated_uncertainty<br/>校准不确定性                      | 0.590 | Annotators prefer answers that are cautious and honest when the visual evidence is ambiguous or insufficient. For example, saying ‘it is unclear what this object is’ is preferred over confidently asserting an unverifiable interpretation. This aligns with human preference for humility in the face of uncertainty. |

**最后的最佳五个criterion:**

| final eval 使用的 best criterion          | score | description                                                  | descrition_zh                                                |
| ----------------------------------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| multimodal_alignment<br/>(多模态一致性)   | 0.625 | The answer should align well with the visual context in terms of spatial, temporal, or semantic relationships. For example, if the question asks about object interactions, the answer should reflect those interactions as shown in the image — not just describe objects in isolation. | 答案应在空间关系、时间关系或语义关系等方面与视觉上下文保持良好一致。例如，如果问题询问的是物体之间的交互关系，那么答案应反映图像中展示的这些交互，而不仅仅是孤立地描述各个物体。 |
| interpretive_fidelity  <br/>解释性忠实度  | 0.696 | Interpretive fidelity is a tie-breaking criterion that applies ONLY when (1) both candidate answers are factually grounded (i.e., neither contains hallucinated visual elements or factual errors), and (2) the visual scene is ambiguous or open to multiple plausible interpretations (e.g., emotional tone, narrative context, symbolic meaning, or behavioral intent). It does NOT apply when: (a) the image is unambiguous or literal (e.g., object count, presence/absence, spatial positioning), (b) the question demands a literal factual answer, or (c) one candidate contains a hallucination — in which case that answer is disqualified and interpretive fidelity is inapplicable. Annotators must first verify factual grounding (via visual alignment) before applying this criterion. If both answers are grounded and ambiguous, interpretive fidelity compares which interpretation is more contextually consistent with visible cues (e.g., facial expression, posture, composition, object arrangement) — even if speculative — as long as it does not contradict visual evidence. For example, 'the person is smiling' is grounded; 'the person is joyful' is interpretive but still valid if the smile is visible. 'The person is arrogant' is valid only if visual cues (e.g., smirk, raised brow) support it — it is not hallucination if grounded in visible cues. If both interpretations are equally grounded and plausible, interpretive fidelity does not resolve the tie — the tie is broken by other criteria (e.g., specificity, completeness, or calibrated uncertainty). Hallucination = invented visual elements not present in the image (e.g., 'the cat is flying' when no wings are visible). Speculation = plausible interpretation supported by visual context (e.g., inferring emotion from expression). Interpretive fidelity does not penalize plausible speculation — only favors the interpretation with stronger visual anchors. It must NEVER be used to override factual grounding — if one answer is hallucinated or factually incorrect, it is disqualified regardless of interpretive plausibility. Examples: 'The cat is near the edge of the pond' > 'The cat is swimming' (if no water visible — the latter is hallucination, so interpretive fidelity does not apply). 'The woman is waiting for her turn to speak' > 'The woman is reading' (if stage cues visible — the former is contextually supported, the latter is less grounded). If both are grounded and ambiguous, 'waiting' wins over 'reading' if stage cues support waiting. If neither is grounded, interpretive fidelity is inapplicable — the case is resolved via visual grounding or factual consistency. | 解释性忠实度是一项仅在以下情况下适用的平局判定标准：(1) 两个候选答案在事实层面均有依据（即都不存在幻觉视觉元素或事实错误），且 (2) 视觉场景具有歧义或存在多种合理解释（例如情绪基调、叙事语境、象征意义或行为意图）。在以下情况中不适用：(a) 图像是明确或字面化的（例如物体数量、存在/不存在、空间位置），(b) 问题要求的是字面事实答案，或 (c) 某一候选答案包含幻觉——在这种情况下该答案应被取消资格，解释性忠实度不适用。标注者必须首先通过视觉对齐验证事实基础，然后才能应用该标准。如果两个答案都具有事实依据且存在歧义，则解释性忠实度比较哪种解释在语境上更符合可见线索（例如面部表情、姿态、构图、物体排列）——即使带有推测性，只要不与视觉证据矛盾。例如，“这个人正在微笑”是有依据的；“这个人很开心”是解释性的，但如果微笑可见仍然有效。“这个人很傲慢”只有在视觉线索（如冷笑、挑眉）支持时才有效——只要基于可见线索，就不属于幻觉。如果两种解释同样有依据且同样合理，解释性忠实度不能解决平局——应由其他标准（如具体性、完整性或校准的不确定性）决定。幻觉 = 图像中不存在的虚构视觉元素（例如“猫在飞”，但没有翅膀）。推测 = 基于视觉语境的合理解释（例如从表情推断情绪）。解释性忠实度不会惩罚合理推测——只会偏好具有更强视觉锚点的解释。它绝不能用于推翻事实基础——如果一个答案是幻觉或事实错误，无论解释多么合理，都应被取消资格。示例：“猫在池塘边缘” > “猫在游泳”（如果没有水体可见——后者是幻觉，因此解释性忠实度不适用）。“女人在等待轮到她发言” > “女人在阅读”（如果有舞台线索——前者更符合语境，后者锚点较弱）。如果两者都有依据且存在歧义，在舞台线索支持“等待”的情况下，“等待”优于“阅读”。如果两者都无依据，解释性忠实度不适用——应通过视觉基础或事实一致性解决。 |
| completeness_of_response<br/>回答的完整性 | 0.600 | The answer should cover all aspects of the question without omitting key information that is visually evident or logically implied. Annotators should evaluate whether the answer fully addresses the intent of the question, especially when multiple objects, actions, or relationships are involved. This criterion is applicable when one answer leaves out critical elements visible in the image that are necessary to fully answer the question | 答案应涵盖问题的所有方面，不遗漏视觉上显而易见或逻辑上隐含的关键信息。标注人员应评估答案是否充分满足问题意图，尤其是在涉及多个对象、动作或关系时。本标准适用于当某个答案遗漏了图像中可见的关键元素，而这些元素对于完整回答问题是必要的情况。 |
| temporal_consistency<br/>时间一致性       | 0.643 | Assesses whether the answer is consistent with the implied temporal sequence or state of the scene — particularly when motion, action, or change over time is involved. This criterion applies when the image contains dynamic elements (e.g., people in motion, moving objects, or implied sequence). Annotators should penalize answers that suggest actions or states that are temporally inconsistent with the visual evidence. Example: If the image shows a person mid-jump, an answer claiming 'the person has just landed' is inconsistent, while 'the person is in mid-air' is temporally accurate. This criterion is not applicable for static images or questions without implied temporal dynamics. | 评估答案是否与场景中隐含的时间顺序或状态一致——尤其是在涉及运动、动作或随时间变化的情况下。该标准适用于图像包含动态元素（例如正在移动的人、运动中的物体或隐含的过程顺序）时。标注者应对那些在时间上与视觉证据不一致的动作或状态描述进行扣分。例如：如果图像显示一个人处于跳跃的空中阶段，而答案称“这个人刚刚落地”，则是不一致的；而“这个人正在空中”则在时间上是准确的。对于静态图像或不涉及隐含时间动态的问题，该标准不适用。 |
| action_grounding<br/>动作定位             | 0.619 | Evaluates whether the answer explicitly and accurately grounds actions or behaviors on observable visual cues — not inferred or imagined. Annotators should check if the answer describes actions that are visibly supported by motion, posture, or interaction cues in the image. This criterion is distinct from visual grounding and focuses specifically on behavioral or action-related claims. Example: If an image shows a person lifting weights, an answer stating 'the person is lifting' is action-grounded, while 'the person is straining' (without visual evidence of strain) is not. This criterion is not applicable when no action is depicted or implied. | 评估答案是否明确且准确地将动作或行为建立在可观察的视觉线索之上，而不是基于推测或想象。标注者应检查答案是否描述了能够通过运动、姿态或互动线索在图像中清晰支持的动作。该标准不同于视觉定位，专门关注与行为或动作相关的陈述。例如：如果图像中显示一个人在举重，那么“这个人在举重”是动作定位正确的，而“这个人很吃力”（没有视觉证据表明其吃力）则不是。当图像中未呈现或未暗示任何动作时，该标准不适用。 |

**实验结果分析**：

- 在所有迭代中，**没有一个是good criterion**。

- 没有**任何初始 criterion 进入最终 eval 集合**。

- **初始 criteria 太泛。**
  visual_grounding、factual_consistency、question_relevance 都是合理的人类偏好原则，但它们覆盖面太大，worker 在具体 pair 上不一定能稳定地区分 A/B。

- 最优的criterion 和初始化的criterion 在验证集上的Valid 结果是一样的，这里没有体现优化的作用

- mid 在refection 之后的 rewrite, description越来越长。如果是重新生成的criterion 不会太长。

- **最终 criteria 更像“专门判别器”。**

  - `interpretive_fidelity` 只在“双方都 factual grounded 且场景有歧义”时作为 tie-breaker；<br/>"Refuse to Respond":  "0.05 (2)"
  - `action_grounding` 专门判 action；"Refuse to Respond": "0.925 (37)"
  - `temporal_consistency `专门判动态/时序；"Refuse to Respond": "0.775 (31)"
  - `completeness_of_response` 专门判多要素遗漏。

  这种细分让 criterion 更容易命中特定错误类型。而且**这些准则覆盖率低，对于很多样本选择拒绝回答**

- **最强信号是 interpretive_fidelity。**
  它从 broad visual grounding 进一步细化为：先排除 hallucination/factual error，再判断哪个解释更贴近可见线索。这个设计比单纯说“要视觉 grounded”更可操作。

- **hallucination 这个初始 criterion 表现最差。**
  avoidance_of_hallucinated_visual_details = 0.442，说明直接用“不要幻觉”做 pairwise criterion 可能太粗，worker 很难稳定应用。是否要划分为更细的子类

- 需要重新修改一下manager的提示词，来提高初始化criterion的质量。同时manager 可以改成更强大的模型