all:
	mkdir -p lib bin
	gcc -shared -fPIC src/nn.c -O2 -Iinclude -lm -o lib/libnn.so
	gcc -c -O2 -o bin/main.o src/main.c -Iinclude
	gcc -O2 bin/main.o -L`pwd`/lib -lnn -Wl,-rpath,`pwd`/lib -o bin/a.out

clean:
	rm -rf lib bin
