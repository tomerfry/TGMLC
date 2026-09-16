all:
	mkdir -p lib bin
	gcc -shared -fPIC src/nn.c -O2 -Iinclude -lm -o lib/libnn.so
	gcc -c -O2 -o bin/adder.o src/adder.c -Iinclude
	gcc -O2 bin/adder.o -L`pwd`/lib -lnn -Wl,-rpath,`pwd`/lib -o bin/adder

clean:
	rm -rf lib bin
