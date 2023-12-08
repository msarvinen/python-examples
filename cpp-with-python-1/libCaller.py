#!/usr/bin/python3

import ctypes
import sys

def main():
    if len(sys.argv) == 3:
        TestLib = ctypes.cdll.LoadLibrary('./stLib.so')
        print(TestLib.SampleAddInt(int(sys.argv[1]), int(sys.argv[2])))

if __name__ == '__main__':
	main()
