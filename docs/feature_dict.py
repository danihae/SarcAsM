import os
import sys
# Ensure the sarcasm module is in the PYTHONPATH
sys.path.insert(0, os.path.abspath('..'))
from sarcasm.features import structure_feature_dict, motion_feature_dict, grouping_feature_dict
from sarcasm.motion import MotionAnalysis
from sarcasm.structure import StructureAnalysis


def get_type_name(data_type):
    """
    Get the name of the data type for display in the table.

    Parameters
    ----------
    data_type : type
        The data type to get the name of.

    Returns
    -------
    str
        The name of the data type.
    """
    if hasattr(data_type, '__origin__'):
        origin = data_type.__origin__.__name__
        args = ', '.join([arg.__name__ if hasattr(arg, '__name__') else str(arg) for arg in data_type.__args__])
        return f'{origin}[{args}]'
    else:
        return data_type.__name__


def _feature_cell(value):
    """The name in bold, then the description; a name with LaTeX ($...$) is written as math, without bold."""
    name = value["name"]
    if "$" in name:
        name = ''.join(f":math:`{part}`" if i % 2 else part for i, part in enumerate(name.split("$")))
    else:
        name = f"**{name}**"
    description = ' '.join(value["description"].split())
    return f"{name}. {description}" if description else name


def _list_table(items, caption, key_width):
    """One list-table: key | feature | data type. The class lets _static/feature_tables.css wrap the text."""
    table = f".. list-table:: {caption}\n"
    table += "   :header-rows: 1\n"
    table += f"   :widths: {key_width} {88 - key_width} 12\n"
    table += "   :class: feature-table\n\n"
    for row in [['Key', 'Feature', 'Data type']] + [
            [f"``{key}``", _feature_cell(value), get_type_name(value["data type"])] for key, value in items]:
        table += "   * - " + "\n     - ".join(row) + "\n"
    return table


def dict_to_list_table(dictionary, underline='-'):
    """The features, one table per function that computes them (one section each if there are several), in the
    order the functions are defined in their class, which is the order of the analysis."""
    classes = {'structure': StructureAnalysis, 'motion': MotionAnalysis}
    by_function = {}
    for key, value in dictionary.items():
        by_function.setdefault(value['function'], []).append((key, value))

    def defined_at(function):
        # 'sarc.structure.analyze_vectors' -> position of analyze_vectors in StructureAnalysis (its bases after it)
        namespace, method = function.split('.')[1:]
        names = [name for klass in classes[namespace].__mro__ for name in vars(klass)]
        return names.index(method) if method in names else len(names)

    # the key column as wide as the longest key needs, the same in all tables of one call
    key_width = min(34, max(18, round(0.66 * max(len(key) for key in dictionary) + 4)))
    out = []
    for function in sorted(by_function, key=defined_at):
        items = by_function[function]
        namespace, method = function.split('.')[1:]
        # link to the class that defines the method (the motion methods live in mixins)
        owner = next((k for k in classes[namespace].__mro__ if method in vars(k)), classes[namespace])
        caption = f"Computed by :py:meth:`{function} <{owner.__module__}.{owner.__qualname__}.{method}>`"
        if len(by_function) > 1:
            out.append(f"{function}\n{underline * len(function)}\n\n")
        out.append(_list_table(items, caption, key_width) + "\n")
    return ''.join(out)


def generate_rst_content(title, table, intro):
    return f"""
.. _{title.lower().replace(' ', '_')}:

{title}
{'=' * len(title)}

{intro}

{table}
"""


# The track and LOI keys are registered with the structure features but belong on the motion page.
structure_only = {k: v for k, v in structure_feature_dict.items() if k.startswith('structure.')}
tracks_and_lois = {k: v for k, v in structure_feature_dict.items() if k.startswith('motion.')}
groupings = {f'motion.groups.<kind>.{leaf}': v for leaf, v in grouping_feature_dict.items()}

table_structure = dict_to_list_table(structure_only)

with open('./structure_features.rst', 'w') as file:
    file.write(generate_rst_content(
        "Feature Reference", table_structure,
        "The following tables describe the structure features analyzed by SarcAsM, one table per "
        "analysis step, keyed by the path they are stored under. A key is its path, so "
        "``sarc.data['structure.sarcomere.oop']`` and "
        "``sarc.data.structure.sarcomere.oop`` are the same value. Print "
        "``sarc.data`` for an overview, ``sarc.data.find('slen')`` to search and "
        "``sarc.data.describe(key)`` for the entry below. The motion features are listed "
        "in :ref:`motion_features`:"))

with open('./motion_features.rst', 'w') as file:
    file.write(generate_rst_content(
        "Motion Features",
        dict_to_list_table(motion_feature_dict),
        "The following table describes the per-group motion features analyzed by "
        "SarcAsM. They are computed by :py:meth:`sarc.motion.analyze() <sarcasm.motion.analysis.AnalysisMixin.analyze>` "
        "and stored under ``motion.<kind>.<feature>``, where ``kind`` is the grouping "
        "(``pool``, ``mband``, ``myofibril``, ``loi``, ``domain`` or ``custom``), "
        "e.g. ``sarc.data['motion.pool.beating_rate']``. Every kind writes the same features, "
        "and a group ``g = sarc.motion.groups(kind)[i]`` reads its row as ``g.beating_rate``:"))
    file.write("""
Tracks and lines of interest
----------------------------

The tracks, one row per tracked sarcomere (``g.tracks.<name>`` reads the members of a group),
and the lines of interest of the ``loi`` grouping:

""" + dict_to_list_table(tracks_and_lois, underline='~') + """

Groupings
---------

Each grouping kind is stored on its own under ``motion.groups.<kind>.*``, so several groupings
live side by side:

""" + dict_to_list_table(groupings, underline='~'))
