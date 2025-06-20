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
    scenario_name: str,
    params_dict: dict[str, any],
    override_params: dict[str, any] = {},
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
        if value is not None:
            if " " in str(value) or "=" in str(value):
                command += f" --{param_name} '{value}'"
            else:
                command += f" --{param_name} {value}"
    return command.strip()


def load_meta_commands(filename: str) -> dict[str, list[MetaCommand]]:
    meta_commands: dict[str, list[MetaCommand]] = {}
    with open(filename, "rb") as f:
        scenario_commands = json.load(f)
        for scenario in scenario_commands:
            if not scenario in meta_commands:
                meta_commands[scenario] = []

            for command in scenario_commands[scenario]:
                meta_commands[scenario].append(MetaCommand(scenario, command))

    return meta_commands


def build_commands(meta_commands_filename: str, commands_folder: str) -> list[str]:
    meta_command_groups = load_meta_commands(meta_commands_filename)
    commands: list[str] = []
    for meta_command_group in meta_command_groups:
        scenario_parameters_filename = os.path.join(
            commands_folder, f"krknctl-input-{meta_command_group}.json"
        )
        if not os.path.exists(scenario_parameters_filename):
            raise Exception(f"{meta_command_group} input file not found")
        with open(scenario_parameters_filename, "rb") as f:
            scenario_parameters = json.load(f)
            params_dict = {p["name"]: p for p in scenario_parameters}
            for meta_command in meta_command_groups[meta_command_group]:
                command = generate_command(
                    meta_command_group, params_dict, meta_command.params
                )

                commands.append(
                    '{"instruction": "%s", "output": "%s"}'
                    % (meta_command.prompt, command)
                )
    return commands
