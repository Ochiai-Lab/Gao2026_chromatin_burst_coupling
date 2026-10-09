"""Read the article's sectioned Source Data workbook without modifying it."""

from __future__ import annotations

from pathlib import Path
import posixpath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def cell_value(cell, shared):
    kind = cell.get('t')
    if kind == 'inlineStr':
        return ''.join(t.text or '' for t in cell.iter(S + 't'))
    node = cell.find(S + 'v')
    if node is None or node.text is None:
        return None
    value = node.text
    if kind == 's':
        return shared[int(value)]
    if kind == 'b':
        return value == '1'
    if kind in ('str', 'e'):
        return value
    number = float(value)
    return int(number) if number.is_integer() else number


def read_sections(path: Path):
    """Keep Excel row identities and the labels of each published data block."""
    tables = []
    with ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in item.iter(S + 't'))
                      for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        targets = {x.get('Id'): x.get('Target') for x in
                   ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        sheets = ET.fromstring(archive.read('xl/workbook.xml')).find(S + 'sheets')
        for sheet in sheets:
            target = targets[sheet.get(R + 'id')]
            part = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            table = None
            with archive.open(part) as stream:
                iterator = ET.iterparse(stream, events=('start', 'end'))
                _, root = next(iterator)
                for event, row in iterator:
                    if event != 'end' or row.tag != S + 'row':
                        continue
                    cells = {c.get('r').rstrip('0123456789'): cell_value(c, shared)
                             for c in row if c.tag == S + 'c'}
                    populated = {k: v for k, v in cells.items() if v is not None}
                    number = int(row.get('r'))
                    if len(populated) == 1 and isinstance(populated.get('A'), str):
                        table = {'sheet': sheet.get('name'), 'title': populated['A'],
                                 'title_row': number, 'header_row': None,
                                 'columns': [], 'column_labels': [], 'rows': []}
                        tables.append(table)
                    elif table is not None and not table['columns'] and len(populated) >= 2:
                        table['columns'] = list(populated.values())
                        table['column_labels'] = list(populated)
                        table['header_row'] = number
                    elif table is not None and table['columns'] and len(populated) >= 2:
                        table['rows'].append({'row': number, 'values': {
                            label: cells.get(column) for label, column in
                            zip(table['columns'], table['column_labels'])}})
                    row.clear()
                    root.clear()
    return tables
