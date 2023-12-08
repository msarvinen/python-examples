#### Shared C library usage in Python
This example demonstrates how to write, buid & compile shared library written in pure C and then used from python code.

#### C-code
Example C-code is shown in stLib.c -file.
1st) C++ library code need to be compiled to object file by using g++:
_g++ -c -o stLib.o stLib.c_

Now you should have stLib.o file in your directory.

2nd) Object file must be converted to shared (.SO) file by using gcc:
_gcc -shared -o stLib.so stLib.o_

Now you should have library file stLib.so in your directory.

#### USAGE:
_python3 libCaller.py 111 222_
