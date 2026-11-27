"""Run the packaged chart adapter using exclusive copies where hard links are unavailable.

Both paths must be inside this deck's private build directory. No input is overwritten;
the packaged adapter still validates all chart data and emits its provenance receipt.
"""
import os
from pathlib import Path
import runpy
import shutil
import sys

build=Path(__file__).resolve().parent/'.build'
def exclusive_copy(source,destination,*args,**kwargs):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if not source.is_relative_to(build) or not destination.is_relative_to(build):
        raise ValueError('Chart adapter paths must stay in the deck build directory.')
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src,dst)
        dst.flush();os.fsync(dst.fileno())

helper=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(helper.parent))
os.link=exclusive_copy
runpy.run_path(str(helper),run_name='__main__')
