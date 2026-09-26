"""Text-only skill rendering. Experiment resources are configured separately."""
from src.infra.artifacts import validate_skill


def attach_prerequisite(skill: dict) -> dict:
    skill = validate_skill(skill)
    framing = skill.get("payload_framing", "")
    return {**skill, "body": (framing + "\n" if framing else "") + skill["body"]}


def render_skill_md(skill: dict) -> str:
    # JSON strings are valid YAML scalar values, including multiline descriptions.
    import json
    return (f"---\nname: {json.dumps(skill['name'])}\n"
            f"description: {json.dumps(skill['description'])}\n---\n\n{skill['body']}\n")
