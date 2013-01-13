#!/usr/bin/env python3

import os
import subprocess
import sys

def compile_executable(name, source, thread_height, thread_width,
                       block_height, block_width, args):
    print('Compiling', name, '...')
    proc = subprocess.Popen(['./tools/compile_executable.sh', name, source,
                             str(thread_height), str(thread_width),
                             str(block_height), str(block_width)] + args)
    ret_code = proc.wait()
    if ret_code != 0:
        print('ERROR: Failed to compile', name, file=sys.stderr)

def remove_executable(name):
    print('Removing', name, '...')
    try:
        os.remove(name)
    except OSError as e:
        pass

    try:
        os.remove(name + '.ini')
    except OSError as e:
        pass

MATRIX_DIMS = [16, 32, 64, 128, 256, 512]

def get_matrix_simple_name(matrix_dim, thread_dim, block_dim):
    return 'matrix_simple_' + \
           str(matrix_dim) + '_' + \
           str(thread_dim) + '_' + \
           str(block_dim)

def build_matrix_simple():
    for matrix_dim in MATRIX_DIMS:
        thread_dim = matrix_dim
        while thread_dim >= 1:
            block_dim = matrix_dim // thread_dim
            name = get_matrix_simple_name(matrix_dim, thread_dim, block_dim)
            compile_executable(name, 'matrix_simple.c',
                               thread_dim, thread_dim,
                               block_dim, block_dim,
                               ['-DSIZE=' + str(matrix_dim)])
            thread_dim //= 2

def clean_matrix_simple():
    for matrix_dim in MATRIX_DIMS:
        thread_dim = matrix_dim
        while thread_dim >= 1:
            block_dim = matrix_dim // thread_dim
            name = get_matrix_simple_name(matrix_dim, thread_dim, block_dim)
            remove_executable(name)
            thread_dim //= 2

def build_helloworld():
    compile_executable('helloworld', 'helloworld.c', 4, 4, 4, 4, [])

def clean_helloworld():
    remove_executable('helloworld')

def make_all():
    build_helloworld()
    build_matrix_simple()

def make_clean():
    clean_helloworld()
    clean_matrix_simple()

if __name__ == '__main__':
    action = 'all'

    try:
        action = sys.argv[1]
    except IndexError as e:
        pass

    if action == 'all':
        make_all()
    elif action == 'clean':
        make_clean()
    else:
        print('ERROR: Unknown action', action, file=sys.stderr)
