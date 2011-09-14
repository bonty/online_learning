OBJS = cw_tool.o cw.a

cw_tool : $(OBJS)
	g++ -o cw_tool $(OBJS)

cw.a : cw.o
	ar r $@ $<
	ranlib $@

.cpp.o :
	g++ -c $< -g -O2

all : cw_tool

clean :
	rm -rf *.o *.a