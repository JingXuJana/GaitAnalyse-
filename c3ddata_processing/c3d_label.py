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


def print_label(reader):
    results = []
    # print('Header information:\n{}'.format(reader.header))
    for high_level_key, g in sorted(reader.group_items()):
        # print('')
        for low_level_key, p in sorted(g.param_items()):
            if high_level_key == 'EVENT' and (low_level_key == 'ICON_IDS' or low_level_key == 'TIMES'):
                result = print_param(g, p)
                results.append(result)
    return results
        


def print_param_value(name, value):
    print(name, '=', value)


def print_param_array(name, p, offset_in_elements):
    ICONID =[]
    label_time = []
    start = offset_in_elements
    end = offset_in_elements + p.dimensions[0]
    if p.bytes_per_element == 2:
        ICONID = p.int16_array
        return ICONID
    
    elif p.bytes_per_element == 4:
        # print("bytes_per_element == 4")
        label_time = p.float_array
        # print(label_time)
        return label_time
    elif p.bytes_per_element == -1:
        # print("p.bytes_per_element == -1")
        return print_param_value(name, p.bytes[start:end])
    else:
        arr = p.int8_array
    # print('{0} = {1}'.format(name, arr.flatten()[start:end]))


def print_param(g, p):
    label_ICON =[]
    label_TIME =[]
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
        label_ICON = print_param_array(name, p, 0)
        

    if len(p.dimensions) >= 2:
       #  print('We are in case 3.')
        offset = 0
        for coordinate in product(*map(range, reversed(p.dimensions[1:]))):
            subscript = ''.join(["[{0}]".format(x) for x in coordinate])
            if subscript == '[0]' and offset == 0 :
                label_TIME = print_param_array(name + subscript, p, offset)
                break 
            # print(f'name is {name}, subscript is {subscript}, offset is {offset}')
            offset += p.dimensions[0]

    return label_ICON,label_TIME
    
    # return label_ICON,label_TIME


def main():
    args = parser.parse_args()
    for filename in args.input:
        try:
            if filename == '-':
                print('*** (stdin) ***')
                print_label(c3d.Reader(sys.stdin))
            else:
                print('*** {} ***'.format(filename))
                with open(filename, 'rb') as handle:
                    print_label(c3d.Reader(handle))
        except Exception as err:
            print(err)


if __name__ == '__main__':
    main()