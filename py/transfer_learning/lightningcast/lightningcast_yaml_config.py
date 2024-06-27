from yaml_cascade import YamlCascade
from typing import Mapping, Tuple, Union, Any, Optional
from pathlib import Path
import copy
import re

from functools import reduce


def flatten_dict(dict_in, starting_key=""):
    dict_out = {}
    for key, value in dict_in.items():

        if starting_key == "":
            key_to_assign = key
        else:
            key_to_assign = f"{starting_key}.{key}"

        if isinstance(value, dict):
            dict_to_append = flatten_dict(value, key_to_assign)
        else:
            dict_to_append = {key_to_assign: value}

        dict_out = {**dict_out, **dict_to_append}

    return dict_out


def unflatten_dict(dict_in):
    out_dict = {}
    for key, value in dict_in.items():
        keys = list(key.split("."))
        base = out_dict
        count = 0
        for key in keys:
            count += 1
            if count == (len(keys)):
                base[key] = value
            else:
                if key in base:
                    base = base[key]
                else:
                    base[key] = {}
                    base = base[key]
    return out_dict


# if the value in brackets points to a litteral we will replace it. If it doesn't then we wont
def replace_value_literal(str_in, reg_patt, **dict_in):
    def replace_worker(match_found):
        capture_group = match_found.group(1)
        if reg_patt.match(dict_in[capture_group]):
            return match_found.group()
        else:
            return dict_in[capture_group]

    return reg_patt.sub(replace_worker, str_in)


# will parse strings and strings in lists
def parse_flat_dict(
    dict_in,
):  # go through values and replace all that have brackets pointing to a literal. If no
    # changes are made on a pass then end fail if there are still brackets or succeed if there aren't.
    out_dict = copy.copy(dict_in)
    reg_patt = re.compile(r"(?<!{){([^{}]+)}(?!})")
    while True:
        changes = 0
        value_combined_string = ""
        for key, value in out_dict.items():
            if isinstance(value, str):
                try:
                    out_dict[key] = replace_value_literal(value, reg_patt, **dict_in)
                except KeyError as e:
                    raise RuntimeError(
                        f"When Parsing LightningCast Yaml Config a key was referenced that doesn't exist: {e}."
                    )
                if out_dict[key] != value:
                    changes += 1
                value_combined_string += f"-{value}-"
            elif isinstance(value, list):
                for x in range(len(value)):
                    if isinstance(value[x], str):
                        value[x] = replace_value_literal(value[x], reg_patt, **dict_in)
                out_dict[key] = value
            else:
                out_dict[key] = value
        dict_in = out_dict
        if changes == 0:
            if re.search(reg_patt, value_combined_string):
                raise RuntimeError(
                    f"When Parsing LightningCast Yaml Config it was detected that some values reference each other recursively!"
                )
            break
    return out_dict


def parse_and_format_dict(work_dict):
    work_dict = copy.copy(work_dict)
    return unflatten_dict(parse_flat_dict(flatten_dict(work_dict)))


class LightningcastYamlCascade(YamlCascade):
    """Lightningcast Yaml Config. Inherits from Ray Garcia's YamlCascade and inspired by his examples of YamlCascade Subclasses.
    This is a YAML cascade (recursive dictionary addressable as compound keys), but this derived class
    tweaks behavour for the lightningcast application. Specifically it lets part of the yaml config access otherparts of the config with: {key}.
    """

    def __init__(
        self,
        default_context: Optional[Mapping],
        first_yaml: Union[dict, str, Path],
        *yaml_paths,
        **substitutions,
    ):
        self.default_context = default_context
        super().__init__(first_yaml, *yaml_paths, **substitutions)
        self._tree = parse_and_format_dict({**self._tree, **default_context})

    # I want values set as Null in the yaml to return None but values that don't exist in the yaml to return an error. This function uses a more complicated function
    # then Ray's reduce - lambda function from the super class. It takes alot more lines then that one line function but it distinguishes betten keys that return None
    # because they are set to null in the yaml and keys that return null because they aren't set.
    def __getitem__(self, item: Any, default: Any = KeyError) -> Any:
        if isinstance(item, str):
            keys = tuple(item.split("."))

        count = 0
        rebuild_key = ""
        working_dict = self._tree

        for key in keys:
            if rebuild_key == "":
                rebuild_key = key
            else:
                rebuild_key = (
                    rebuild_key + "." + key
                )  # We use this so we can pinpoint directly where things go wrong and which part of the key is invalid.
            count += 1
            if count == len(keys):
                if not key in working_dict:
                    if default is KeyError:
                        raise KeyError(f"{rebuild_key} key is unknown")
                    else:
                        return default
                else:
                    return working_dict[key]
            try:
                working_dict = working_dict[key]
            except KeyError as e:  # We really shouldn't need to except an error here. Either of the other two raise statements should have happened instead. But I'll leave this incase I missed an edge case.
                raise KeyError(f"{e} in {rebuild_key} key is unkown")
            if not isinstance(working_dict, dict):
                raise KeyError(f"{rebuild_key} doesn't resolve to a dictionary type")

    def get(self, item: Any, default: Any = KeyError) -> Any:
        return self.__getitem__(item, default)
