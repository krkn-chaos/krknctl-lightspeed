import json
import os.path


class MetaCommand:
    def __init__(self, scenario_name: str, params: dict[str, any]):
        self.scenario_name = scenario_name
        self.prompt = params["prompt"] if "prompt" in params else ""
        self.params = params["params"] if "params" in params else {}

    scenario_name: str
    prompt: str
    params: dict[str, str | int]


def generate_command(
    scenario_name: str, params_dict: dict[str, any], override_params={}
) -> str:

    command = f"krknctl run {scenario_name}"

    sorted_param_names = sorted(params_dict.keys())

    for param_name in sorted_param_names:
        param_info = params_dict[param_name]

        value = override_params.get(param_name)

        if value is None:
            if "default" in param_info and param_info["default"] != "":
                value = param_info["default"]
            elif param_info.get("required") == "true":
                pass
            else:
                continue

        if param_info.get("type") == "boolean":
            if value == True:
                command += f" --{param_name}"
            else:
                continue

        if value is not None:
            if " " in str(value) or "=" in str(value):
                command += f" --{param_name} '{value}'"
            else:
                command += f" --{param_name} {value}"

    return command.strip()


def load_meta_commands(filename: str) -> dict[str, list[MetaCommand]]:
    meta_commands: dict[str,list[MetaCommand]] = {}
    with open(filename, "rb") as f:
        scenario_commands = json.load(f)
        for scenario in scenario_commands:
            if not scenario in meta_commands:
                meta_commands[scenario] = []

            for command in scenario_commands[scenario]:
                meta_commands[scenario].append(MetaCommand(scenario, command))

    return meta_commands


def build_commands(training_filename: str, meta_commands_filename: str, commands_folder: str) -> list[str]:
    commands: list[str] = []
    meta_commands = load_meta_commands(meta_commands_filename)
    for command in meta_commands:
        commands: list[str] = []

    with open(os.path.join(commands_folder,f"krknctl-input-{command}.json"), "w", encoding="utf-8") as f:
        loaded_commands = json.load(f)
        for entry in commands:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
