import os
import sys
# Ensure the sarcasm module is in the PYTHONPATH
sys.path.insert(0, os.path.abspath('..'))
from sarcasm.features import structure_feature_dict, motion_feature_dict


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


def dict_to_list_table(dictionary, caption='Table of features'):
    header = ['Feature (dict key)', 'Name', 'Function', 'Description', 'Data Type']
    rows = []
    for key, value in dictionary.items():
        function_link = f":py:func:`sarcasm.{value['function']}`"
        data_type = get_type_name(value["data type"])
        description = value["description"].replace('\n', ' ').replace('  ', ' ')

        # Handle LaTeX in the 'name' value, both fully LaTeX and mixed content
        name = value["name"]
        if "$" in name:
            parts = name.split("$")
            processed_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    processed_parts.append(f":math:`{part}`")
                else:
                    processed_parts.append(part)
            name = ''.join(processed_parts)

        row = [f"`{key}`", name, function_link, description, data_type]
        rows.append(row)

    table = f".. list-table:: {caption}\n"
    table += "   :header-rows: 1\n"
    table += "   :widths: 15 15 30 28 12\n\n"

    # Add header
    table += "   * - " + "\n     - ".join(header) + "\n"

    # Add rows
    for row in rows:
        table += "   * - " + "\n     - ".join(row) + "\n"

    return table


def generate_rst_content(title, table, intro):
    return f"""
.. _{title.lower().replace(' ', '_')}:

{title}
{'=' * len(title)}

{intro}

{table}
"""


table_structure = dict_to_list_table(structure_feature_dict, 'Table of features by key')
table_motion = dict_to_list_table(motion_feature_dict, 'Table of per-group motion features')

with open('./structure_features.rst', 'w') as file:
    file.write(generate_rst_content(
        "Feature Reference", table_structure,
        "The following table describes the features analyzed by SarcAsM, keyed by "
        "the path they are stored under. A key is its path, so "
        "``sarc.data['structure.sarcomere.oop']`` and "
        "``sarc.data.structure.sarcomere.oop`` are the same value. Print "
        "``sarc.data`` for an overview, ``sarc.data.find('slen')`` to search and "
        "``sarc.data.describe(key)`` for the entry below:"))

with open('./motion_features.rst', 'w') as file:
    file.write(generate_rst_content(
        "Motion Features", table_motion,
        "The following table describes the per-group motion features analyzed by "
        "SarcAsM. They are computed by :py:meth:`sarcasm.SarcAsM.analyze_track_motion` "
        "and stored under ``motion.<kind>.<feature>``, where ``kind`` is the grouping "
        "used (``pool``, ``mband``, ``myofibril``, ``domain``, ``loi`` or ``custom``) "
        "— e.g. ``sarc.data['motion.pool.beating_rate']``, equivalently "
        "``sarc.data.motion.pool.beating_rate``. Every kind writes the same members:"))