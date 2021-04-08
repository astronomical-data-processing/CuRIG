CC=nvcc
LDFLAGS+= -lcufft
output = output
input = nufft.cu
all: generate
generate:
    $(CC) -o $(output) $(input)  $(LDFLAGS)