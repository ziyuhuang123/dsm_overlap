CUTLASS_DIR=${PWD}/../external/cutlass
CXX=nvcc
APP=transpose

# CXXFLAGS=-DNDEBUG --generate-code=arch=compute_90a,code=[compute_90a] -std=c++17 -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include --expt-relaxed-constexpr

CXXFLAGS=-std=c++17 -DNDEBUG -gencode arch=compute_90a,code=sm_90a -lineinfo -Xptxas=--verbose -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include --expt-relaxed-constexpr

LDFLAGS=

LDLIBS=-lcuda

OBJECTS = main.o 

.SUFFIXES: .o .cu

default: clean $(APP)

$(APP): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

main.o:
	$(CXX) -c $(CXXFLAGS) -o "$@" main.cu

python:
	CUTE_TRANSPOSE_DIR=${PWD} CUTLASS_DIR=${CUTLASS_DIR} pip3 install python/

clean: 
	rm -f $(OBJECTS) $(APP)
