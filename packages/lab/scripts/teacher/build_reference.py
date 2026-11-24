"""Rebuild the shipped reference optima: python scripts/teacher/build_reference.py --task 9.2 --task 9.3"""
import sys

from mlab.teacher.sweep import main

if __name__ == "__main__":
    sys.exit(main())
