"""Initialize S0: case signatures -> semantic clusters -> complete children."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from structured_rubrics.structured.schema import StructuredRubric

from . import rubric_pipeline as pipeline
from .experiment_utils import atomic_write_json as write, load_json
from .manager_runtime import Manager, nonempty
from .model_call_support import file_sha256


# SIGNATURE_SYSTEM_PROMPT = """你是负责初始化结构化 Rubric 的 Manager。

# 本阶段的任务是：分析一个具体案例中人类选择背后的比较依据，并提炼与目标根相关、可复用的偏好判断模式。

# 全部根准则用于理解不同维度的职责，本次围绕目标根进行分析。
# 结合图像、问题和两个候选回答，找出能够解释人类选择的关键差异，分析人类可能更看重哪些因素，以及如何权衡两个回答的优缺点。

# 参考目标 Worker 的报告，理解当前判断关注了哪些证据，是否遗漏了重要差异，或采用了不同的比较依据。
# Worker 与人类偏好不一致，本身不足以说明目标根的判断错误；
# 整体选择可能同时受到多个维度的影响。

# basis 应说明两个回答的具体差异，以及这些证据如何支持对人类选择的解释。
# signature 应概括模式出现的情形、影响偏好的关键因素及其比较关系。
# 如果没有发现目标根下有证据支持、值得提炼的模式，返回 applicable=false，并说明原因。

# 只返回一个 JSON 对象：
# {
#     "applicable": false/true,
#     "basis": "两个候选回答之间的具体差异，以及这些差异如何帮助解释人类的偏好选择",
#     "signature": "可复用的偏好判断模式，包括其适用情境、关键偏好因素，以及在比较过程中如何权衡这些因素"
# }
# """


# SIGNATURE_USER_TEMPLATE = """## 全部根准则
# 以下是当前 Rubric 的全部根准则，用于理解各维度的职责：
# {roots_context}

# ## 本次分析的目标根
# 根 ID：{target_root_id}
# 名称：{target_root_name}
# 职责：{target_root_description}

# ## 案例
# 样本 ID：{sample_id}

# 原始问题：
# {question}

# 候选回答 A：
# {candidate_a}

# 候选回答 B：
# {candidate_b}

# 该案例的图像随本消息附上。

# ## 人类偏好
# 该案例的整体人类偏好为：{gold}

# 请结合图像、问题和回答内容，分析哪些关键差异能够解释这一选择。

# ## 目标 Worker 的判断报告
# 以下报告来自目标根对应的 Worker，供你参考当前判断所使用的证据与比较依据。

# 对回答 A 的分析：
# {worker_analysis_a}

# 对回答 B 的分析：
# {worker_analysis_b}

# 比较理由：
# {worker_thought}

# 局部判断结果：
# {worker_answer}

# 请围绕目标根的职责，提炼这个案例中可复用的偏好判断模式，并用具体证据说明其与人类选择的关系。
# """


# CLUSTER_SYSTEM_PROMPT = """你是负责初始化结构化 Rubric 的 Manager。

# 本阶段的任务是：将多个案例提取出的 signature，归纳为目标根下共同的偏好判断模式。

# 参考全部根准则的职责分工，寻找不同案例中反复出现的比较依据：
# 在什么情形下，哪些回答特征影响人类选择，以及人类如何权衡这些特征。

# 按照共同的判断情形和比较依据进行分组，不要仅依据相似措辞分组。
# 每个 cluster 的 pattern 应清楚表达组内 signature 的共性，为下一阶段生成子准则提供依据。
# 无法形成一致模式的 signature 可以不分配。

# 目标是形成 2–5 个有证据支持的 cluster。
# 每个 cluster 至少包含两个不同的 signature ID，
# 同一个 ID 不得出现在多个 cluster 中。
# 无法形成一致模式的 signature 可以不分配。不要为了凑数拆分或编造。

# 只返回一个 JSON 对象：
# {
#     "clusters": [
#         {
#             "pattern": "共同的适用情形、偏好因素及比较依据",
#             "signature_ids": ["S001", "S002"]
#         }
#     ],
#     "unassigned_ids": ["S003"]
# }
# """


# CLUSTER_USER_TEMPLATE = """## 全部根准则
# 以下根准则用于理解职责分工：
# {roots_context}

# ## 本次归纳的目标根
# 根 ID：{target_root_id}
# 名称：{target_root_name}
# 职责：{target_root_description}

# ## 待归纳的 signature
# 以下 signature 来自前一阶段对具体案例中人类偏好的分析。
# 每条记录包含 signature ID、偏好判断模式和案例依据：
# {signatures_context}

# 请寻找这些案例中共同的偏好依据与取舍方式，给出各 cluster 的模式描述及其对应的 signature ID。
# """


# CHILDREN_SYSTEM_PROMPT = """你是负责初始化结构化 Rubric 的 Manager。

# 本阶段的任务是：将归纳出的偏好模式，转化为目标根的一组完整、可复用的子准则，帮助 Worker 更好地复原人类比较回答时的判断依据。

# 结合 signature 的案例证据和 cluster 的共同模式，生成 2–5 条简洁、不冗余的子准则。
# 每条子准则围绕一个偏好判断要点，包含：
# 1. 在什么情况下适用；
# 2. 需要检查哪些证据；
# 3. 如何据此判断哪个回答更好。

# 名称应简洁地概括主要偏好判断要点。
# 涉及多个优缺点时，应说明哪些因素影响比较，以及如何进行取舍。

# 参考全部根准则的职责分工，保持子准则属于目标根。
# 参考本轮前面已经生成的子准则，明确目标根对偏好判断的贡献，并保持本组子准则可由对应 Worker 独立使用。
# 不要将具体样本答案、样本 ID 或固定 A/B 偏好写入准则。

# 只返回一个 JSON 对象：
# {
#     "children": [
#         {
#             "name": "简短且组内唯一的 snake_case 名称",
#             "description": "适用条件、检查内容及比较依据"
#         }
#     ],
#     "change_summary": "如何将共同偏好模式转化为子准则"
# }
# """


# CHILDREN_USER_TEMPLATE = """## 全部根准则
# 以下根准则用于理解各维度的职责：
# {roots_context}

# ## 本轮前面已经生成的子准则
# 以下子准则在本轮初始化的前面阶段生成：
# {previous_children_context}

# ## 本次生成子准则的目标根
# 根 ID：{target_root_id}
# 名称：{target_root_name}
# 职责：{target_root_description}

# ## 原始 signature 及案例依据
# 以下材料描述具体案例中的偏好判断模式及其证据：
# {signatures_context}

# ## 归纳出的 cluster
# 以下材料描述共同的偏好依据与取舍方式，以及对应的 signature ID：
# {clusters_context}

# 请将这些偏好模式转化为目标根的完整子准则组，同时说明这些子准则如何体现归纳出的共同偏好模式。
# """


SIGNATURE_SYSTEM_PROMPT = """You are the Manager responsible for initializing a structured Rubric.

Your task at this stage is to analyze the grounds for the human choice in a specific case and extract reusable preference judgment patterns relevant to the target root.

Use all root criteria to understand the responsibilities of different dimensions, and focus this analysis on the target root.
Using the image, question, and two candidate responses, identify the key differences that can explain the human choice. Analyze which factors humans may value more and how they weigh the strengths and weaknesses of the two responses.

Refer to the target Worker's report to understand which evidence the current judgment considered, whether it missed important differences, or whether it used different grounds for comparison.
Disagreement between the Worker and human preference does not, by itself, establish that the target root's judgment is wrong;
the overall choice may be influenced by multiple dimensions.

basis should describe the specific differences between the two responses and how this evidence supports an explanation of the human choice.
signature should summarize the circumstances in which the pattern arises, the key factors affecting preference, and how those factors relate in the comparison.
If you find no evidence-supported pattern worth extracting under the target root, return applicable=false and explain why.

Return only one JSON object. The applicable field must be true or false:
{
    "applicable": false/true,
    "basis": "Specific differences between the two candidate responses and how these differences help explain the human preference choice",
    "signature": "Reusable preference judgment pattern, including its applicable context, key preference factors, and how to weigh these factors in the comparison"
}
"""


SIGNATURE_USER_TEMPLATE = """## All root criteria
The following are all root criteria in the current Rubric. Use them to understand the responsibilities of each dimension:
{roots_context}

## Target root for this analysis
Root ID: {target_root_id}
Name: {target_root_name}
Responsibility: {target_root_description}

## Case
Sample ID: {sample_id}

Original question:
{question}

Candidate response A:
{candidate_a}

Candidate response B:
{candidate_b}

The image for this case is attached to this message.

## Human preference
The overall human preference for this case is: {gold}

Using the image, question, and response content, analyze which key differences can explain this choice.

## Target Worker's judgment report
The following report comes from the Worker assigned to the target root. Use it as a reference for the evidence and grounds for comparison used in the current judgment.

Analysis of response A:
{worker_analysis_a}

Analysis of response B:
{worker_analysis_b}

Reasoning for the comparison:
{worker_thought}

Local judgment:
{worker_answer}

Within the target root's responsibilities, extract reusable preference judgment patterns from this case and use specific evidence to explain their relationship to the human choice.
"""


CLUSTER_SYSTEM_PROMPT = """You are the Manager responsible for initializing a structured Rubric.

Your task at this stage is to group signatures extracted from multiple cases into shared preference judgment patterns under the target root.

Refer to the responsibilities of all root criteria and look for recurring grounds for comparison across cases:
under what circumstances, which response features affect the human choice, and how humans weigh those features.

Group signatures by shared judgment situations and grounds for comparison, rather than by similar wording alone.
Each cluster's pattern should clearly express what its signatures have in common and provide a basis for generating child criteria in the next stage.
Signatures that do not form a coherent pattern may remain unassigned.

Aim to form 2–5 clusters supported by evidence.
Each cluster must contain at least two distinct signature IDs,
and no ID may appear in more than one cluster.
Signatures that do not form a coherent pattern may remain unassigned. Do not split or invent patterns merely to meet a count.

Return only one JSON object:
{
    "clusters": [
        {
            "pattern": "Shared applicable situations, preference factors, and grounds for comparison",
            "signature_ids": ["S001", "S002"]
        }
    ],
    "unassigned_ids": ["S003"]
}
"""


CLUSTER_USER_TEMPLATE = """## All root criteria
Use the following root criteria to understand their respective responsibilities:
{roots_context}

## Target root for this grouping
Root ID: {target_root_id}
Name: {target_root_name}
Responsibility: {target_root_description}

## Signatures to group
The following signatures come from analyses of human preference in specific cases in the previous stage.
Each record contains a signature ID, a preference judgment pattern, and case evidence:
{signatures_context}

Look for shared preference considerations and trade-offs across these cases, and provide a pattern description and the corresponding signature IDs for each cluster.
"""


CHILDREN_SYSTEM_PROMPT = """You are the Manager responsible for initializing a structured Rubric.

Your task at this stage is to turn the grouped preference patterns into a complete set of reusable child criteria for the target root, helping the Worker better reconstruct the grounds humans use when comparing responses.

Combine the case evidence in the signatures with the shared patterns in the clusters to generate 2–5 concise, nonredundant child criteria.
Each child criterion should focus on one key point in judging preference and include:
1. When it applies;
2. What evidence to examine;
3. How to use that evidence to judge which response is better.

The name should concisely summarize the main point in judging preference.
When multiple strengths and weaknesses are involved, explain which factors affect the comparison and how to weigh them.

Refer to the responsibilities of all root criteria and keep the child criteria within the target root's scope.
Consider the previously generated child criteria when identifying this root's contribution to preference judgment. Keep its child group independently usable by its Worker.
Do not encode specific sample answers, sample IDs, or a fixed preference for A or B in the criteria.

Return only one JSON object:
{
    "children": [
        {
            "name": "Short snake_case name unique within the group",
            "description": "Applicability, checks, and grounds for comparison"
        }
    ],
    "change_summary": "How shared preference patterns were turned into child criteria"
}
"""


CHILDREN_USER_TEMPLATE = """## All root criteria
Use the following root criteria to understand the responsibilities of each dimension:
{roots_context}

## Previously generated child criteria
The following child criteria were generated earlier in this initialization:
{previous_children_context}

## Target root for child generation
Root ID: {target_root_id}
Name: {target_root_name}
Responsibility: {target_root_description}

## Original signatures and case evidence
The following material describes preference judgment patterns in specific cases and their supporting evidence:
{signatures_context}

## Grouped clusters
The following material describes shared preference considerations and trade-offs, along with the corresponding signature IDs:
{clusters_context}

Turn these preference patterns into a complete child criterion group for the target root, and explain how the child criteria reflect the shared preference patterns identified.
"""


PROMPTS = dict(signature=SIGNATURE_SYSTEM_PROMPT, cluster=CLUSTER_SYSTEM_PROMPT,
               children=CHILDREN_SYSTEM_PROMPT)
USER_TEMPLATES = dict(signature=SIGNATURE_USER_TEMPLATE, cluster=CLUSTER_USER_TEMPLATE,
                      children=CHILDREN_USER_TEMPLATE)
PROMPT_VERSION = "init-split-template-en-v3-sequential-children"


def render_user_prompt(stage, payload):
    """Render descriptions around original text; never format the inserted text."""
    root = payload["root"]
    fields = dict(
        roots_context="\n\n".join(
            f"Root ID: {item['root_id']}\nName: {item['name']}\nResponsibility: {item['description']}"
            for item in payload["roots"]),
        target_root_id=root["root_id"], target_root_name=root["name"],
        target_root_description=root["description"],
        previous_children_context="\n\n".join(
            f"### {item['root_id']}: {item['name']}\n"
            + "\n\n".join(
                f"Name: {child['name']}\nDescription: {child['description']}"
                for child in item["children"])
            for item in payload.get("previous_children", [])
        ) or "No child criteria have been generated yet.",
    )
    if stage == "signature":
        case = payload["case"]
        worker = case["worker"]
        fields.update(
            sample_id=case["sample_id"], question=case["question"],
            candidate_a=case["A"], candidate_b=case["B"], gold=case["gold"],
            worker_analysis_a=worker["analysis_a"], worker_analysis_b=worker["analysis_b"],
            worker_thought=worker["thought"], worker_answer=worker["answer"],
        )
    else:
        fields["signatures_context"] = "\n\n".join(
            f"### {item['signature_id']}\nPreference pattern: {item['signature']}\nCase evidence: {item['basis']}"
            for item in payload["signatures"])
        if stage == "children":
            clusters = payload["clusters"]
            fields["clusters_context"] = "\n\n".join(
                f"### Cluster {index}\nShared preference pattern: {item['pattern']}\n"
                f"Signature IDs: {', '.join(item['signature_ids'])}"
                for index, item in enumerate(clusters["clusters"], 1))
            fields["clusters_context"] += (
                "\n\nUnassigned signature IDs: " + ", ".join(clusters.get("unassigned_ids", [])))
    return USER_TEMPLATES[stage].format(**fields)


def validate(stage, result, payload):
    """Keep the original applicability and cluster structure requirements."""
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    if stage == "signature":
        if type(result.get("applicable")) is not bool:
            raise ValueError("applicable must be a boolean")
        nonempty(result.get("basis"), "basis")
        if result["applicable"]:
            nonempty(result.get("signature"), "signature")
    elif stage == "cluster":
        clusters = result.get("clusters")
        if not isinstance(clusters, list) or not 0 <= len(clusters) <= 5:
            raise ValueError("need a list of at most 5 supported clusters")
        allowed = {s["signature_id"] for s in payload["signatures"]}
        seen = set()
        for cluster in clusters:
            nonempty(cluster.get("pattern"), "pattern")
            ids = cluster.get("signature_ids")
            if (not isinstance(ids, list) or len(set(ids)) < 2
                    or len(set(ids)) != len(ids) or not set(ids) <= allowed
                    or seen.intersection(ids)):
                raise ValueError("clusters need >=2 unique known IDs, without overlap")
            seen.update(ids)
        result["unassigned_ids"] = sorted(allowed - seen)
    elif stage == "children":
        return pipeline.validate_children(result)
    else:
        raise ValueError(f"unknown initialization stage: {stage}")
    return result


def make_manager(config, attempts=10, client=None):
    return Manager(config, attempts, client, prompts=PROMPTS, validator=validate)


def case_payload(row, record, root):
    if record["order"] != 0:
        raise ValueError("initialization expects frozen Discovery K=1 original order")
    case = {key: row[key] for key in ("sample_id", "question", "A", "B")}
    worker = record["subtrees"][root]["parsed"]
    case.update(gold=row["answer"], worker={
        key: worker[key] for key in ("analysis_a", "analysis_b", "thought", "answer")})
    return case


def project_previous_children(rubric, groups):
    return pipeline.project_rubric(pipeline.replace_groups(rubric, groups), groups)


def signatures(manager, directory, root, rubric, rows, current):
    records = pipeline.system_records(current)

    def one(row):
        payload = dict(roots=pipeline.project_rubric(rubric),
                       root=pipeline.project_rubric(rubric, [root])[0],
                       case=case_payload(row, records[row["sample_id"]], root))
        result = manager.call(
            "signature", directory / f"{row['_signature_id']}.json", payload, [row],
            user_text=render_user_prompt("signature", payload))
        return dict(signature_id=row["_signature_id"], sample_id=row["sample_id"], **result)

    return pipeline.parallel(rows, one, manager.config.get("stage_concurrency", {}).get(
        "signature", manager.config["concurrency"]))


def generate_children(manager, directory, root, rubric, library, clusters, previous_groups):
    payload = dict(
        roots=pipeline.project_rubric(rubric),
        root=pipeline.project_rubric(rubric, [root])[0],
        signatures=[s for s in library if s["applicable"]], clusters=clusters,
        previous_children=project_previous_children(rubric, previous_groups),
    )
    proposal = manager.call("children", directory / "children.json", payload,
                            user_text=render_user_prompt("children", payload))
    pipeline.replace_groups(rubric, {root: proposal["children"]})
    return proposal


def generate_group(manager, directory, root, rubric, library, previous_groups=None):
    selected = [s for s in library if s["applicable"]]
    if len(selected) < 4:
        return None
    payload = dict(roots=pipeline.project_rubric(rubric),
                   root=pipeline.project_rubric(rubric, [root])[0], signatures=selected)
    clusters = manager.call("cluster", directory / "clusters.json", payload,
                            user_text=render_user_prompt("cluster", payload))
    if len(clusters["clusters"]) < 2:
        return None
    proposal = generate_children(manager, directory, root, rubric, library,
                                 clusters, previous_groups or {})
    return proposal, clusters


def reuse_source(config, target, source):
    """Copy matching saved R0/discovery reports; never write to the source run."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("source run and new output must be different directories")
    source_config = load_json(source / "run_config.json")
    for key in ("model", "base_url", "request_kwargs"):
        if config["manager"][key] != source_config["manager"][key]:
            raise ValueError(f"source Manager {key} differs")
    rows = pipeline.load_rows(config, "discovery")
    source_rows = pipeline.load_rows(source_config, "discovery")

    def content(items):
        return [{key: row[key] for key in ("sample_id", "question", "A", "B", "answer")}
                | {"image_sha256": file_sha256(Path(row["image_path"]))} for row in items]

    if content(rows) != content(source_rows):
        raise ValueError("source discovery cases or A/B mapping differs")
    r0 = StructuredRubric.load_json(source / "r0/rubric.json")
    value = load_json(source / "r0/system.json")
    if (value["rubric_sha256"] != r0.rubric_sha256 or value["k"] != 1
            or [s["sample_id"] for s in value["samples"]] != [r["sample_id"] for r in rows]
            or any(s["replicates"]["0"]["order"] != 0 for s in value["samples"])
            or value["metrics"]["technical_failure_count"]
            or value["metrics"] != pipeline.system.metrics(value, rows)):
        raise ValueError("source R0 discovery reports differ")
    metadata = dict(source_run=str(source), r0_sha256=r0.rubric_sha256,
                    source_files={name: file_sha256(source / name)
                                  for name in ("r0/rubric.json", "r0/system.json",
                                               "init/rubric.json", "vlrb/r0.json",
                                               "vlrb/initial.json")
                                  if (source / name).is_file()})
    provenance = target / "source.json"
    if provenance.exists() and load_json(provenance) != metadata:
        raise ValueError("source run changed; use a new output directory")
    for name in ("r0/rubric.json", "r0/system.json"):
        destination = target / name
        if destination.exists() and file_sha256(destination) != file_sha256(source / name):
            raise ValueError(f"reused R0 artifact changed at {destination}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
    write(provenance, metadata)
    print(f"reused frozen R0 and Discovery K=1 reports: {r0.rubric_sha256}", flush=True)
    return r0


def initialize(config, target, rows, manager=None, attempts=10, r0=None,
               reuse_patterns_from=None):
    """Create S0 and an epoch-zero state; leave evolution explicitly incomplete."""
    started = time.perf_counter()
    r0 = pipeline.build_multicrit_open_ended_init_rubric() if r0 is None else r0
    manager = manager or make_manager(
        dict(config["manager"], env_file=config.get("env_file", ".env")), attempts)
    protocol = dict(version=PROMPT_VERSION, system_prompts=PROMPTS,
                    user_templates=USER_TEMPLATES)
    source_roots = None
    if reuse_patterns_from is not None:
        reuse_patterns_from = Path(reuse_patterns_from).resolve()
        source_summary = load_json(reuse_patterns_from / "init/summary.json")
        source_roots = {item["root_id"]: item for item in source_summary["roots"]}
        protocol["reuse_patterns_from"] = str(reuse_patterns_from)
    path = target / "init/protocol.json"
    if path.exists() and load_json(path) != protocol:
        raise ValueError("Init Split prompts changed; use a new output directory")
    write(path, protocol)
    for i, row in enumerate(rows, 1):
        row["_signature_id"] = f"S{i:03d}"
    write(target / "r0/rubric.json", r0.to_dict())
    baseline = pipeline.evaluate(config, target, "r0/system", rows, r0, attempts=attempts)
    records = pipeline.system_records(baseline)
    groups, diagnostics = {}, []
    for i, root in enumerate(r0.root_ids, 1):
        directory = target / f"init/r{i:02d}"
        primary = [r for r in rows if
                   records[r["sample_id"]]["subtrees"][root]["parsed"]["answer"] != r["answer"]]
        print(f"init root={root}: primary cases={len(primary)}", flush=True)
        if source_roots is not None:
            source_root = source_roots[root]
            library = load_json(reuse_patterns_from / f"init/r{i:02d}/library.json")
            clusters = source_root["clusters"]
            write(directory / "reused/clusters.json", clusters)
            proposal = generate_children(manager, directory / "reused", root, r0,
                                         library, clusters, groups)
            generated = proposal, clusters
            expanded_count = source_root["expanded_count"]
            print(f"init root={root}: reused signatures and clusters", flush=True)
        else:
            library = signatures(manager, directory / "signatures", root, r0, primary, baseline)
            folder = directory / "primary"
            generated = generate_group(manager, folder, root, r0, library, groups)
            extra = []
            if generated is None:
                seen = {r["sample_id"] for r in primary}
                extra = [r for r in rows if r["sample_id"] not in seen]
                print(f"init root={root}: supplement cases={len(extra)}", flush=True)
                library += signatures(manager, directory / "signatures", root, r0, extra, baseline)
                folder = directory / "expanded"
                generated = generate_group(manager, folder, root, r0, library, groups)
            expanded_count = len(extra)
        write(directory / "library.json", library)
        if generated is None:
            raise RuntimeError(f"{root}: insufficient supported patterns for two initial clusters; S0 not created")
        proposal, clusters = generated
        groups[root] = proposal["children"]
        diagnostics.append(dict(
            root_id=root, primary_count=len(primary), expanded_count=expanded_count,
            signature_count=len(library), applicable_count=sum(s["applicable"] for s in library),
            clusters=clusters, children_count=len(proposal["children"]),
            description_characters=sum(len(c["description"]) for c in proposal["children"])))
        print(f"init root={root}: applicable={diagnostics[-1]['applicable_count']}/"
              f"{len(library)}, clusters={len(clusters['clusters'])}, "
              f"children={len(proposal['children'])}", flush=True)
    s0 = pipeline.replace_groups(r0, groups)
    write(target / "init/rubric.json", s0.to_dict())
    initial_system = pipeline.evaluate(config, target, "init/system", rows, s0, attempts=attempts)
    if not (target / "state.json").exists():
        write(target / "state.json", dict(epoch=0, rubric=s0.to_dict(),
              baseline="init/system", completed=False, stop_reason=None))
    summary = target / "init/summary.json"
    if not summary.exists():
        write(summary, dict(prompt_version=PROMPT_VERSION, r0_sha256=r0.rubric_sha256,
                            s0_sha256=s0.rubric_sha256, roots=diagnostics,
                            wall_seconds=time.perf_counter() - started,
                            finished_at=datetime.now(timezone.utc).isoformat(),
                            manager_cost=pipeline.manager_cost([target / "init"])))
    print(f"Init Split complete: {target / 'init/rubric.json'}", flush=True)
    return s0, initial_system
