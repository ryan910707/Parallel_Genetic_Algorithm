NVCXX = nvc++
NVCXXFLAGS = -O0 -std=c++14 -fast -acc -gpu=cc60 -Minfo=accel 
TARGETS = OpenAcc

.PHONY: all
all: $(TARGETS)

%: %.cpp
	$(NVCXX) $(NVCXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
