nn:
	mkdir -p lib bin
	gcc -shared -fPIC src/nn.c -O2 -Iinclude -lm -o lib/libnn.so

adder: nn
	gcc -shared -fPIC src/nn.c -O2 -Iinclude -lm -o lib/libnn.so
	gcc -c -O2 -o bin/adder.o src/adder.c -Iinclude
	gcc -O2 bin/adder.o -L`pwd`/lib -lnn -Wl,-rpath,`pwd`/lib -o bin/adder

mlp: nn 
	gcc -shared -fPIC src/nn.c -O2 -Iinclude -lm -o lib/libnn.so
	gcc -c -O2 -o bin/MLP.o src/MLP.c -Iinclude
	gcc -O2 bin/MLP.o -L`pwd`/lib -lnn -Wl,-rpath,`pwd`/lib -o bin/MLP

all: adder mlp

clean:
	rm -rf lib bin
