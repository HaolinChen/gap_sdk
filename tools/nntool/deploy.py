#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2020  GreenWaves Technologies, SAS

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import collections
from filecmp import cmp, cmpfiles
import hashlib
import os
import shutil
import zlib

import argcomplete

from utils.gitignore_parser import gen_match, parse_rules, parse_rules_str


def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='deploy')
    parser.add_argument('files', nargs=2, help='in and out files')
    parser.add_argument('-n', '--no_changes', action='store_true', help="don't make any changes")
    parser.add_argument('-v', '--verbose', action='store_true', help="print actions")
    return parser

BLOCK_SIZE = 4096

def md5_chunk(chunk):
    """
    Returns md5 checksum for chunk
    """
    m = hashlib.md5()
    m.update(chunk)
    return m.hexdigest()


def adler32_chunk(chunk):
    """
    Returns adler32 checksum for chunk
    """
    return zlib.adler32(chunk)

# Checksum objects
# ----------------
Signature = collections.namedtuple('Signature', 'md5 adler32')


class Chunks(object):
    def __init__(self):
        self.chunks = []
        self.chunk_sigs = {}

    def append(self, sig):
        self.chunks.append(sig)
        self.chunk_sigs.setdefault(sig.adler32, {})
        self.chunk_sigs[sig.adler32][sig.md5] = len(self.chunks) - 1

    def get_chunk(self, chunk):
        adler32 = self.chunk_sigs.get(adler32_chunk(chunk))

        if adler32:
            return adler32.get(md5_chunk(chunk))

        return None

    def __getitem__(self, idx):
        return self.chunks[idx]

    def __len__(self):
        return len(self.chunks)


def checksums_file(fn):
    chunks = Chunks()
    with open(fn) as f:
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk:
                break

            chunks.append(
                Signature(
                    adler32=adler32_chunk(chunk),
                    md5=md5_chunk(chunk)
                )
            )

        return chunks

def _get_block_list(file_one, file_two):
    checksums = checksums_file(file_two)
    blocks = []
    offset = 0
    with open(file_one, 'rb') as f:
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk:
                break

            chunk_number = checksums.get_chunk(chunk)

            if chunk_number is not None:
                offset += BLOCK_SIZE
                blocks.append(chunk_number)
                continue
            else:
                offset += 1
                blocks.append(chunk[0])
                f.seek(offset)
                continue

    return blocks

def file(file_one, file_two, args):
    if not os.path.exists(file_two):
        if args.verbose:
            print(f'create {file_two}')
        if not args.no_changes:
            if os.path.isfile(file_one):
                shutil.copyfile(file_one, file_two)
            else:
                shutil.copytree(file_one, file_two)
        return
    if not cmp(file_one, file_two):
        if args.verbose:
            print(f'write {file_two}')
        if not args.no_changes:
            with open(file_two, 'wb') as ft:
                shutil.copyfile(file_one, file_two)

RULES_FILES = [
    '.gitignore',
    '.deployignore'
]

def gen_rules(dir_path, rules):
    new_rules = []
    for filename in RULES_FILES:
        new_rules.extend(parse_rules(os.path.join(dir_path, filename)))
    new_rules.extend(parse_rules_str('.git*\n.deploy*', dir_path))
    return rules + new_rules

def read_dir(from_path, to_path, rules, args):
    rules = gen_rules(from_path, rules)
    match = gen_match(rules)
    file_list_from = set()
    for filename in os.listdir(from_path):
        full_path_from = os.path.join(from_path, filename)
        if match(full_path_from):
            if args.verbose:
                print(f'skip {full_path_from}')
            continue
        file_list_from.add(filename)
        full_path_to = os.path.join(to_path, filename)
        if os.path.isdir(full_path_from):
            if not os.path.exists(full_path_to):
                if args.verbose:
                    print(f'make dir {full_path_to}')
                if not args.no_changes:
                    os.mkdir(full_path_to)
            elif not os.path.isdir(full_path_to):
                if args.verbose:
                    print(f'change to dir {full_path_to}')
                if not args.no_changes:
                    os.remove(full_path_to)
                    os.mkdir(full_path_to)

            read_dir(full_path_from, full_path_to, rules, args)
        else:
            file(full_path_from, full_path_to, args)
    if args.no_changes and not os.path.exists(to_path):
        return
    for filename in os.listdir(to_path):
        if filename in file_list_from:
            continue
        full_path_to = os.path.join(to_path, filename)
        if args.verbose:
            print(f'delete {full_path_to}')
        if not args.no_changes:
            if os.path.isdir(full_path_to):
                shutil.rmtree(full_path_to)
            else:
                os.remove(full_path_to)


def main():
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    read_dir(os.path.abspath(args.files[0]), os.path.abspath(args.files[1]), [], args)


if __name__ == '__main__':
    main()
