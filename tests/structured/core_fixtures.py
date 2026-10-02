"""Small offline artifacts shared by current-method tests."""

import json

from experiments.evolving_structured_rubrics import aligned_system_runtime as system


def rows(n=6):
    return [dict(sample_id=f"sample-{i}", question="Compare image content", A="candidate-a",
                 B="candidate-b", answer="A", image_path="image.png", source=f"source-{i%3}",
                 domain="visual", _signature_id=f"S{i+1:03d}") for i in range(n)]


def report(answer="A", thought="visible detail supports this choice"):
    return dict(analysis_a="A evidence", analysis_b="B evidence", thought=thought, answer=answer)


def artifact(data, rubric, correct, *, k=1):
    samples = []
    for i, row in enumerate(data):
        replicas = {}
        for rep in range(k):
            replicas[str(rep)] = dict(order=0, subtrees={
                r: dict(parse_ok=True, parsed=report("B"), raw_response=json.dumps(report("B")))
                for r in rubric.root_ids},
                arbiter=dict(parse_ok=True, parsed=report("A" if i < correct else "B"),
                             raw_response=json.dumps(report("A" if i < correct else "B"))))
        samples.append(dict(sample_id=row["sample_id"], replicates=replicas))
    value = dict(k=k, samples=samples, rubric_sha256=rubric.rubric_sha256,
                 protocol_version=system.PROTOCOL_VERSION)
    value["metrics"] = system.metrics(value, data)
    return value
