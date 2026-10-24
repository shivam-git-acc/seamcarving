NVCC ?= nvcc
ARCH ?= native

seam_cuda: seam_cuda.cu stb_image.h stb_image_write.h
	$(NVCC) -O3 -arch=$(ARCH) -Xcompiler -fopenmp -lgomp -diag-suppress 550 -o $@ $<

stb_image.h stb_image_write.h:
	./scripts/get_deps.sh

test:
	g++ -O2 -std=c++17 -o tests/dp_emulation_test tests/dp_emulation_test.cpp
	./tests/dp_emulation_test

clean:
	rm -f seam_cuda tests/dp_emulation_test output.jpg out.png *.nsys-rep *.sqlite *.ncu-rep

.PHONY: test clean
