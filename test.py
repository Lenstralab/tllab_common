#!/usr/bin/python

import sys
import os
import numpy as np
from pathlib import Path
from tllab_common.wimread import imread
from tllab_common.findcells import findcells
from tiffwrite import IJTiffWriter

fname = Path(__file__)
test_files = Path(fname).parent / 'test_files'
wimread = test_files / 'wimread'


# This file defines tests to be run to assert the correct working of our scripts
# after updates. Add a test below as a function, name starting with 'test', and
# optionally using 'assert'.
#
# Place extra files used for these tests in the folder test_files, add imports
# above this text.
#
# Then navigate to the directory containing this file and run ./test.py directly
# from the terminal. If you see red text then something is wrong and you need to
# fix the code before committing to gitlab.
#
# wp@tl20200124


def test_findcell_a(tmp_path):
    with imread(test_files / 'findcell.a.tif') as a:
        c, n = findcells(a(0), a(1), ccdist=150, thres=1, removeborders=True)
        assert np.all(c == a(2)), 'Cellmask wrong'
        assert np.all(n == a(3)), 'Nucleusmask wrong'
    files = [tmp_path / f for f in ('cell.tif', 'nucleus.tif')]
    with IJTiffWriter(files, (1, 1, 1)) as tif:
        for i, f in enumerate((c, n)):
            tif.save(i, f, 0, 0, 0)

    for file, f in zip(files, (c, n)):
        with imread(file) as im:
            assert np.all(im(0) == f), 'data not stored correctly'


def test_cziread_elyra():
    with imread(wimread / 'cziread' / 'YTL639_2020_06_03__16_56_51.czi') as im:
        assert im.shape == (256, 256, 2, 1, 160)


def test_czi_read_airy():
    with imread(wimread / 'cziread' / 'MK022_del111_1-01-Airyscan Processing-09-Scene-1-P1.czi') as im:
        assert im.shape == (499, 496, 1, 15, 210)


def test_seqread():
    with imread(wimread / 'seqread' / 'YTL985F4-1_30mingal_1' / 'Pos0') as im:
        assert im.shape == (1024, 1024, 2, 9, 2)


def test_metaread():
    with imread(wimread / 'metaread' / 'B110B137_H3H1A1_day0_zstack_18102022_.nd' / 'Pos1') as im:
        assert im.shape == (946, 677, 1, 20, 2)


# ----- This part runs the tests -----
if __name__ == '__main__':
    if len(sys.argv) < 2:
        py = ['3.10']
    else:
        py = sys.argv[1:]

    for p in py:
        print('Testing using python {}'.format(p))
        os.system('python{} -m pytest -n=12 -p no:warnings --verbose {}'.format(p, fname))
        print('')

    imread.kill_vm()
