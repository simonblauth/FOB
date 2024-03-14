from pathlib import Path
from typing import Any, Iterable
import re
import yaml

class YAMLParser():
    def __init__(self) -> None:
        pass

    def parse_yaml(self, file: Path) -> Any:
        """
        Opens and parses a YAML file.
        """
        with open(file, "r", encoding="utf8") as f:
            return yaml.safe_load(f)

    def parse_args_into_searchspace(self, searchspace: dict[str, Any], args: Iterable[str]):
        """
        Overwrites args given in the form of 'this.that=something'. Also supports lists: 'this.that[0]=something'
        """
        for arg in args:
            self._parse_arg_into_searchspace(searchspace, arg)

    def _parse_arg_into_searchspace(self, searchspace: dict[str, Any], arg: str):
        keys, value = arg.split("=")
        keys = keys.split(".")
        keys_with_list_indices = []
        for key in keys:
            match = re.search(r"^(.*?)\[(\-?\d+)\]$", key)
            if match:
                keys_with_list_indices.append(match.group(1))
                keys_with_list_indices.append(int(match.group(2)))
            else:
                keys_with_list_indices.append(key)
        target = searchspace
        for key in keys_with_list_indices[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys_with_list_indices[-1]] = yaml.safe_load(value)

    def merge_dicts_hierarchical(self, lo: dict, hi: dict):
        """
        Overwrites values in `lo` with values from `hi` if they are present in both/
        """
        for k, v in hi.items():
            if isinstance(v, dict) and isinstance(lo.get(k, None), dict):
                self.merge_dicts_hierarchical(lo[k], v)
            else:
                lo[k] = v
