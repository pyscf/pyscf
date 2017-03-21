
include arch.inc
sinclude libs.inc

OBJS = \
  m_io.o \
  m_log.o \
  m_warn.o \
  m_siesta_wfsx.o \
  m_siesta_hsx.o \
  m_precision.o \
  m_die.o \
  m_color.o

all: $(OBJS)
	$(LINK) $(LFLAGS) --shared $(OBJS) $(USER_LIBS) -o libnao.so

clean:
	rm $(OBJS) *.so

%.o:%.F90 arch.inc
	$(TIME) $(FORTRAN) $(FLAGS) $(INCDIR) -c $<

