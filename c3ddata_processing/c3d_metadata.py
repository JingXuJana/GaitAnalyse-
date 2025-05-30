#!/usr/bin/env python

'''Display C3D group and parameter information.'''

from __future__ import print_function
from itertools import product

import c3d
import argparse
import sys

parser = argparse.ArgumentParser(description='Display C3D group and parameter information.')
parser.add_argument('input', default='-', metavar='FILE', nargs='+',
                help='process C3D data from this input FILE')


def print_metadata(reader):
    # print('Header information:\n{}'.format(reader.header))
    for high_level_key, g in sorted(reader.group_items()):
        # print('')
        for low_level_key, p in sorted(g.param_items()):
            if high_level_key == 'POINT' and low_level_key == 'LABELS':
                # print(f'g is: {g} \n')
                # print(f'p is: {p} \n')
                start_index, finish_index,another_index = print_param(g, p)
            
    return start_index, finish_index,another_index
            
# def print_metadata(reader):
#     print('Header information:\n{}'.format(reader.header))
#     for key, g in sorted(reader.group_items()):
#         print('')
#         for key, p in sorted(g.param_items()):
#             print_param(g, p)


def print_param_value(name, value):
    print(name, '=', value)


def print_param_array(name, subscript, p, offset_in_elements):
    arr = []
    start = offset_in_elements
    end = offset_in_elements + p.dimensions[0]
    if p.bytes_per_element == 2:
        arr = p.int16_array
    elif p.bytes_per_element == 4:
        arr = p.float_array
    elif p.bytes_per_element == -1:
        value = p.bytes[start:end]
        if value == b'LHipAngles                    ':
            index = int(subscript[1:-1])
            name = 'start'
        elif value == b'RFootProgressAngles           ':
            index = int(subscript[1:-1])
            name = 'finish'
        elif value == b'LThoraxAngles                 ':
            index = int(subscript[1:-1])
            name = 'another_finish'
        else:
            name = None
            index = 0
        return name, index
        # return print_param_value(name+subscript, value)
    else:
        arr = p.int8_array
    # print('{0} = {1}'.format(name+subscript, value))


def print_param(g, p):
    name = "{0.name}.{1.name}".format(g, p)
    # print('{0}: {1.total_bytes}B {1.dimensions}'.format(name, p))

    if len(p.dimensions) == 0:
        # print('We are in case 1.')
        val = None
        width = len(p.bytes)
        if width == 2:
            val = p.int16_value
        elif width == 4:
            val = p.float_value
        else:
            val = p.int8_value
        print_param_value(name, val)

    if len(p.dimensions) == 1 and p.dimensions[0] > 0:
        # print('We are in case 2.')
        return print_param_array(name, p, 0)

    if len(p.dimensions) >= 2:
        # print('We are in case 3.')
        offset = 0
        for coordinate in product(*map(range, reversed(p.dimensions[1:]))):
            subscript = ''.join(["[{0}]".format(x) for x in coordinate])
            # print(f'name is {name}, subscript is {subscript}, offset is {offset}')
            index_name, index = print_param_array(name, subscript, p, offset)
            # print(f'index_name: {index_name}, index: {index}')
            if index_name == 'start':
                start_index = index
            elif index_name == 'finish':
                finish_index = index
            elif index_name == 'another_finish':
                another_index = index

            offset += p.dimensions[0]         
    return start_index, finish_index, another_index


def main():
    args = parser.parse_args()
    for filename in args.input:
        try:
            if filename == '-':
                # print('*** (stdin) ***')
                print_metadata(c3d.Reader(sys.stdin))
            else:
                # print('*** {} ***'.format(filename))
                with open(filename, 'rb') as handle:
                    print_metadata(c3d.Reader(handle))
        except Exception as err:
            print(err)


if __name__ == '__main__':
    main()