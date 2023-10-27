# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with custom formatter with changed way of displaying
subcommands information.

Changes:
* subcommands are always displayed right after the main command
* removed list of subcommands from description -- they are already listed
  and described
* usage: list subcommands in [...|...] instead of {...|...}
"""

import argparse
import re
from gettext import gettext


# TODO: try to print mutually exclusive groups
class Formatter(argparse.RawDescriptionHelpFormatter):
    """
    Formatter for argparse help.
    """

    def _format_action(self, action):
        # determine the required width and the entry label
        help_position = min(
            self._action_max_length + 2, self._max_help_position
        )
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        padding = " " * self._current_indent
        # no help; start on same line and add a final newline
        if not action.help:
            action_header = f"{padding}{action_header}\n"

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            action_header = f"{padding}{action_header: <{action_width}}  "
            indent_first = 0

        # long action name; start on the next line
        else:
            action_header = f"{padding}{action_header}\n"
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append(f"{' ' * indent_first}{help_lines[0]}\n")
                for line in help_lines[1:]:
                    parts.append(f"{' ' * help_position}{line}\n")

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith("\n"):
            parts.append("\n")

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # new: remove redundant list of subcommands
        if isinstance(action, argparse._SubParsersAction):
            parts.pop(0)
        # return a single string
        return self._join_parts(parts)

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choice_strs = [str(choice) for choice in action.choices]
            # new: add different wrapping for subcommands
            if not isinstance(action, argparse._SubParsersAction):
                result = f"{{{','.join(choice_strs)}}}"
            else:
                result = f"[{'|'.join(choice_strs)}]"
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result,) * tuple_size

        return format

    def _format_actions_usage(self, actions, groups):
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            if not group._group_actions:
                raise ValueError(f"empty group {group}")

            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                group_action_count = len(group._group_actions)
                end = start + group_action_count
                if actions[start:end] == group._group_actions:
                    suppressed_actions_count = 0
                    for action in group._group_actions:
                        group_actions.add(action)
                        if action.help is argparse.SUPPRESS:
                            suppressed_actions_count += 1

                    exposed_actions_count = (
                        group_action_count - suppressed_actions_count
                    )

                    if not group.required:
                        if start in inserts:
                            inserts[start] += " ["
                        else:
                            inserts[start] = "["
                        if end in inserts:
                            inserts[end] += "]"
                        else:
                            inserts[end] = "]"
                    elif exposed_actions_count > 1:
                        if start in inserts:
                            inserts[start] += " ("
                        else:
                            inserts[start] = "("
                        if end in inserts:
                            inserts[end] += ")"
                        else:
                            inserts[end] = ")"
                    for i in range(start + 1, end):
                        inserts[i] = "|"

        # collect all actions format strings
        parts = []
        append_dots = False
        for i, action in enumerate(actions):
            # suppressed arguments are marked with None
            # remove | separators for suppressed arguments
            if action.help is argparse.SUPPRESS:
                parts.append(None)
                if inserts.get(i) == "|":
                    inserts.pop(i)
                elif inserts.get(i + 1) == "|":
                    inserts.pop(i + 1)

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == "[" and part[-1] == "]":
                        part = part[1:-1]

                # add the action string to the list
                # new: place subcommands at the beginning
                if isinstance(action, argparse._SubParsersAction):
                    append_dots = True
                    command, dots = part.split(" ")
                    parts.insert(0, command)
                else:
                    parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = action.format_usage()

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = f"{option_string} {args_string}"

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = f"[{part}]"

                # add the action string to the list
                parts.append(part)

        # insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # new: make sure dots are at the end of usage
        if append_dots:
            parts.append(dots)

        # join all the action items with spaces
        text = " ".join([item for item in parts if item is not None])

        # clean up separators for mutually exclusive groups
        open = r"[\[(]"
        close = r"[\])]"
        text = re.sub(f"({open}) ", r"\1", text)
        text = re.sub(f" ({close})", r"\1", text)
        text = re.sub(f"{open} *{close}", r"", text)
        text = text.strip()

        # return the text
        return text

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = gettext("usage: ")

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = str(self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = str(self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(optionals + positionals, groups)
            usage = " ".join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:
                # break usage into wrappable parts
                part_regexp = r"\(.*?\)+(?=\s|$)|" r"\[.*?\]+(?=\s|$)|" r"\S+"
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = re.findall(part_regexp, opt_usage)
                pos_parts = re.findall(part_regexp, pos_usage)
                assert " ".join(opt_parts) == opt_usage
                assert " ".join(pos_parts) == pos_usage

                # new: if only subcommands are in pos_parts append them
                # to opt_parts, so subcommands are always at the beginning
                if opt_parts and len(pos_parts) == 2 and pos_parts[1] == "...":
                    opt_parts = [pos_parts[0], *opt_parts, pos_parts[1]]
                    pos_parts = []

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = len(indent) - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + " ".join(line))
                            line = []
                            line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + " ".join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent) :]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = " " * (len(prefix) + len(prog) + 1)
                    if opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = " " * len(prefix)
                    parts = opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(pos_parts, indent))
                        lines.extend(get_lines(opt_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = "\n".join(lines)

        # prefix with 'usage:'
        return f"{prefix}{usage}\n\n"
