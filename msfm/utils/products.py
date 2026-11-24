# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen (with Claude)

Helpers shared by the product apps, msfm.apps.products.desy6 and the downstream ones (deep_lss.apps.products.desy6).
A product app derives files and config values one step at a time:

- a config value it derives sits on a line marked "<key>: <value> # <- step values: <dotted.key>", which
  write_marked_values rewrites, refusing any other change to the file, or with check only reports stale values;
- with --provenance, clean_commit refuses uncommitted code and record_provenance keeps one record per step: its
  command, config, commits, inputs (size, mtime) and outputs (sha256).
"""

import copy
import datetime
import hashlib
import os
import re
import subprocess

import numpy as np
import yaml

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

# a config line whose value a step writes: "<key>: <value> # <- step values: <dotted.key>"
VALUE_MARKER = re.compile(
    r"^(?P<head>\s*(?P<key>[A-Za-z_]\w*):\s*)(?P<value>[^#]*?)(?P<gap>\s*)"
    r"(?P<marker>#\s*<-\s*step values:\s*(?P<dotted>[A-Za-z_][\w.]*)\s*)$"
)


def rel_path(path, repo_dir):
    """Repository-relative path for files inside the repository, absolute otherwise."""
    rel = os.path.relpath(os.path.abspath(path), repo_dir)
    return path if rel.startswith("..") else rel


def dry_run(step, inputs, outputs, rel=os.path.relpath):
    """Lists what a step would read and write. Returns None, so nothing is recorded."""
    LOGGER.info(f"dry run of step {step}, nothing is written")
    for path in inputs:
        LOGGER.info(f"  reads  {rel(path)} {'(exists)' if os.path.exists(path) else '(MISSING)'}")
    for path in outputs:
        LOGGER.info(f"  writes {rel(path)} {'(exists, would be replaced)' if os.path.exists(path) else ''}")
    return None


def dotted_get(tree, dotted):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def dotted_set(tree, dotted, value):
    *parents, leaf = dotted.split(".")
    for part in parents:
        tree = tree[part]
    tree[leaf] = value


def yaml_number(x):
    # PyYAML only parses a float with a dot in the mantissa: "1e-05" would be read as a string
    text = repr(x)
    mantissa, _, exponent = text.partition("e")
    return f"{mantissa}.0e{exponent}" if exponent and "." not in mantissa else text


def yaml_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(yaml_number(x) for x in value) + "]"
    return yaml_number(value)


def marked_keys(conf_path):
    """Returns the set of dotted config keys whose lines carry a value marker in conf_path."""
    with open(conf_path) as f:
        return {match["dotted"] for match in map(VALUE_MARKER.match, f.read().splitlines()) if match is not None}


def write_marked_values(conf, conf_path, values, check, rel=os.path.relpath):
    """Writes values (dotted config key -> number or per-bin list) into the marked lines of the config.

    Every marked line needs a value and every value a marked line. The rewritten config must parse to the old one with
    only the marked values replaced. With check, writes nothing and exits with 1 if a marked value is stale.

    Args:
        conf (dict): The parsed content of conf_path.
        conf_path (str): The config file to rewrite.
        values (dict): Dotted config key -> number or list of numbers.
        check (bool): Write nothing, exit with 1 if a marked value is stale.
        rel (callable, optional): Path -> the form it takes in the messages.

    Returns:
        bool: whether the config was written
    """
    rel_conf = rel(conf_path)
    for dotted, value in values.items():
        assert np.all(np.isfinite(value)), f"{dotted}: {value} is not finite"
    with open(conf_path) as f:
        lines = f.read().splitlines(keepends=True)

    new_lines, marked, stale = [], set(), []
    for i_line, line in enumerate(lines, start=1):
        body = line.rstrip("\r\n")
        match = VALUE_MARKER.match(body)
        if match is None:
            # comment-only lines (the header explains the marker) are no markers
            if "step values:" in body and not body.lstrip().startswith("#"):
                raise ValueError(f"{rel_conf}:{i_line}: malformed value marker: {body.strip()}")
            new_lines.append(line)
            continue

        dotted = match["dotted"]
        if dotted.split(".")[-1] != match["key"]:
            raise ValueError(f"{rel_conf}:{i_line}: the marker {dotted} sits on the key {match['key']}")
        if dotted in marked:
            raise ValueError(f"{rel_conf}:{i_line}: {dotted} is marked twice")
        if dotted not in values:
            raise KeyError(f"{rel_conf}:{i_line}: step values computes no {dotted}")
        marked.add(dotted)

        new_body = match["head"] + yaml_value(values[dotted]) + (match["gap"] or " ") + match["marker"]
        if new_body != body:
            stale.append(dotted)
        new_lines.append(new_body + line[len(body) :])
    unmarked = sorted(set(values) - marked)
    if unmarked:
        raise ValueError(f"{rel_conf} has no line marked '# <- step values: <dotted.key>' for {unmarked}")

    # the result parses, each marker sat on the line of its key, and no other value changed
    new_text = "".join(new_lines)
    new_conf = yaml.safe_load(new_text)
    reference = copy.deepcopy(conf)
    for dotted, value in values.items():
        if dotted_get(new_conf, dotted) != value:
            raise ValueError(f"{dotted}: the rewritten config reads {dotted_get(new_conf, dotted)!r}, not {value!r}")
        dotted_set(reference, dotted, value)
    assert new_conf == reference, "the rewrite changed a config value that carries no marker"

    if check:
        for dotted in stale:
            LOGGER.error(f"{rel_conf}: {dotted} = {dotted_get(conf, dotted)!r}, step values: {values[dotted]!r}")
        if stale:
            raise SystemExit(1)
        LOGGER.info(f"{rel_conf}: all {len(values)} marked values are up to date")
        return False

    if not stale:
        # nothing written; whether that still gets a provenance record is the caller's choice
        LOGGER.info(f"{rel_conf}: all {len(values)} marked values are up to date")
        return False
    with open(conf_path, "w") as f:
        f.write(new_text)
    LOGGER.info(f"{rel_conf}: wrote {len(stale)} of {len(values)} marked values: {stale}")
    return True


def git(repo_dir, *cmd):
    return subprocess.run(["git", "-C", repo_dir, *cmd], capture_output=True, text=True, check=True).stdout.strip()


def clean_commit(repo_dir, code_paths):
    """Commit of the checkout at repo_dir. Refuses uncommitted changes in code_paths, so the commit is what runs."""
    changes = git(repo_dir, "status", "--porcelain", "--", *code_paths)
    if changes:
        raise SystemExit(f"commit {' and '.join(code_paths)} in {repo_dir} before a step writes products:\n{changes}")
    return git(repo_dir, "rev-parse", "HEAD")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat(timespec="seconds")


def provenance_record(command, config, commits, inputs, outputs, rel=os.path.relpath):
    """One step's record: when, how, from which code, what it read (size, mtime) and what it wrote (sha256).

    Args:
        command (str): The command line of the step.
        config (str): The config the step ran on, as it should appear in the record.
        commits (dict): Record key -> commit, e.g. {"msfm_commit": <sha>}, in the order they should appear.
        inputs (list): Paths the step read.
        outputs (list): Paths the step wrote.
        rel (callable, optional): Path -> the form it takes in the record.

    Returns:
        dict: The record.
    """
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "config": config,
        **commits,
        "inputs": [{"path": rel(p), "bytes": os.path.getsize(p), "mtime": mtime(p)} for p in inputs],
        "outputs": [{"path": rel(p), "sha256": sha256(p)} for p in outputs],
    }


def record_provenance(prov_path, step, record, step_order, about):
    """Replaces the record of step in the provenance file, ordered by step_order. One record per step; git keeps the
    history.

    Args:
        prov_path (str): The provenance yaml.
        step (str): The step the record belongs to.
        record (dict): The record, from provenance_record.
        step_order (list): The order of the steps in the file; steps outside it follow.
        about (str): What the file is, written at its top.
    """
    steps = {}
    if os.path.isfile(prov_path):
        with open(prov_path) as f:
            steps = (yaml.safe_load(f) or {}).get("steps", {})
    steps[step] = record

    prov = {"about": about, "steps": {name: steps[name] for name in [*step_order, *steps] if name in steps}}
    with open(prov_path, "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, width=119)
