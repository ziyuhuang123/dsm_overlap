CUTLASS_DIR=/root/zyhuang/pure_cutlass/cutlass-main
REPO_DIR=..
CXX=nvcc
APP=main

CXXFLAGS=--generate-code=arch=compute_90a,code=[compute_90a] -std=c++17 -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${REPO_DIR}/include/utils --expt-relaxed-constexpr

LDFLAGS=

LDLIBS=-lcuda

OBJECTS = main.o 

.SUFFIXES: .o .cu

default: clean $(APP)

$(APP): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

main.o:
	$(CXX) -c $(CXXFLAGS) -o "$@" main.cu

clean: 
	rm -f $(OBJECTS) $(APP)
